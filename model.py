"""
Our models are available on Huggingface named TinyStories-1M/3M/9M/28M/33M/1Layer/2Layer 
and TinyStories-Instruct-∗. We use GPT-Neo architecture with window size 256 and context length 512. We use GPT-Neo tokenizer but only keep the top 10K most common tokens.
"""
#GPT Neo is a decoder only model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer
import os
from transformers import AutoConfig, AutoModelForCausalLM

#config file
def load_config(config_file: str):
    class GPTNeoConfig:
        pass

    config = GPTNeoConfig()

    # Configuration specification: key -> type converter
    CONFIG_SPEC = {
        # Model parameters
        'vocab_size': int,
        'max_position_embeddings': int,
        'n_layer': int,
        'n_head': int,
        'n_embd': int,
        'embed_dropout': float,
        'attention_dropout': float,
        'resid_dropout': float,
        'layer_norm_epsilon': float,
        'initializer_range': float,
        'window_size': int,
        'local_heads': int,
        'trained_model_path': str,
        # Training parameters
        'batch_size': int,
        'block_size': int,
        'learning_rate': float,
        'num_epochs': int,
        # Dataset parameters
        'tokenizer_name': str,
        'use_data_fraction': float,
    }

    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments
            
            # Split key and value while allowing for colons in values
            if ':' not in line:
                continue  # Skip malformed lines
                
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key in CONFIG_SPEC:
                # Convert value using specified type and handle errors
                try:
                    setattr(config, key, CONFIG_SPEC[key](value))
                except ValueError as e:
                    raise ValueError(f"Invalid value for {key}: {value}") from e
            else:
                print(f"Warning: Unknown config key '{key}' - skipping")

    return config


class GPTNeoSelfAttention(nn.Module):
    """
    Splits the total heads into some local (windowed) heads and some global (full) heads,
    then merges them back. Each subset still does causal masking.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.local_heads = config.local_heads
        self.global_heads = self.num_heads - self.local_heads

        # dimension per head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = config.window_size

        # Q, K, V projection
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        """
        hidden_states shape: [batch_size, seq_len, embed_dim]
        """
        B, S, E = hidden_states.shape

        # 1) Project Q,K,V
        q = self.q_proj(hidden_states)  # [B, S, E]
        k = self.k_proj(hidden_states)  # [B, S, E]
        v = self.v_proj(hidden_states)  # [B, S, E]

        # 2) Reshape into (B, n_head, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, S, hd]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale by sqrt(d_k)
        q = q * self.scale

        # 3) Split heads into "global" subset and "local" subset
        q_global, q_local = q[:, :self.global_heads], q[:, self.global_heads:]
        k_global, k_local = k[:, :self.global_heads], k[:, self.global_heads:]
        v_global, v_local = v[:, :self.global_heads], v[:, self.global_heads:]

        #
        # -- Global attention block --
        #
        # Full causal mask for global heads
        attn_scores_global = torch.matmul(q_global, k_global.transpose(-1, -2)) 
        # shape: [B, global_heads, S, S]

        # Build the standard causal mask: no attending to future tokens
        causal_mask = torch.ones((S, S), device=hidden_states.device, dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # True above the diagonal

        # Convert True -> -inf
        attn_scores_global = attn_scores_global.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_weights_global = F.softmax(attn_scores_global, dim=-1)
        attn_weights_global = self.attn_dropout(attn_weights_global)  # [B, global_heads, S, S]

        # Multiply by V
        attn_output_global = torch.matmul(attn_weights_global, v_global) 
        # shape: [B, global_heads, S, head_dim]

        #
        # -- Local attention block --
        #
        # We do a smaller causal mask that only keeps [i - window_size : i]
        # and sets everything else to -inf.
        attn_scores_local = torch.matmul(q_local, k_local.transpose(-1, -2)) 
        # shape: [B, local_heads, S, S]

        # Build local+causal mask
        # We want to *unmask* only positions within window_size to the left (and i itself),
        # but remain causal (no looking forward).
        full_mask = torch.ones((S, S), dtype=torch.bool, device=hidden_states.device)
        for i in range(S):
            low = max(0, i - self.window_size)
            full_mask[i, low:i+1] = False  # mark these as not masked
        # True in full_mask => "should be masked"
        attn_scores_local = attn_scores_local.masked_fill(full_mask, float('-inf'))

        attn_weights_local = F.softmax(attn_scores_local, dim=-1)
        attn_weights_local = self.attn_dropout(attn_weights_local)

        attn_output_local = torch.matmul(attn_weights_local, v_local) 
        # shape: [B, local_heads, S, head_dim]

        
        # 4) Combine global + local outputs back along the "head" dimension
        #
        attn_output = torch.cat([attn_output_global, attn_output_local], dim=1)  
        # shape now [B, n_head (=16), S, head_dim]

        # 5) Re‑merge heads: (B, n_head, S, head_dim) -> (B, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)

        # 6) Final projection + residual dropout
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


#
# EXAMPLE OF USING IT IN A BLOCK
#

class NewGELUActivation(nn.Module):

    #Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    #the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415


    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))



class GPTNeoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expand 4x in the hidden dimension, then project back
        inner_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner_dim)
        self.c_proj = nn.Linear(inner_dim, config.n_embd)
        self.act = NewGELUActivation()  # or "new" GELU
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTNeoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(config)

    def forward(self, x):
        # attention sub-block
        a = self.ln_1(x)
        a = self.attn(a)
        x = x + a
        # MLP sub-block
        m = self.ln_2(x)
        m = self.mlp(m)
        x = x + m
        return x


class GPTNeoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.embed_dropout)

        self.h = nn.ModuleList([GPTNeoBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None):
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTNeoForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPTNeoModel(config)
        # Create an lm_head that *does not* duplicate its own weight parameter:
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights: the final projection reuses the input embedding matrix
        self.lm_head.weight = self.transformer.wte.weight
        # no additional parameters are created!

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.transformer(input_ids, position_ids)
        logits = self.lm_head(hidden_states)
        return logits


def initialize_weights(module, config):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def generate_text(model, tokenizer, prompt, max_new_tokens=50, 
                  temperature=1.0, top_k=0, device='cpu'):
    """
    Generate text from a (trained) GPTNeoForCausalLM model using 
    an iterative token-by-token approach.

    Args:
        model (GPTNeoForCausalLM): Your causal LM model.
        tokenizer: The tokenizer corresponding to your model.
        prompt (str): The initial text prompt to condition on.
        max_new_tokens (int): How many new tokens to generate after the prompt.
        temperature (float): Scales logits before sampling (higher = more random).
        top_k (int): If > 0, restrict sampling to top-k most probable tokens at each step.
        device (str): "cpu" or "cuda".

    Returns:
        str: The generated text (prompt + newly generated tokens).
    """
    model.eval()
    model.to(device)

    # 1) Encode the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    # We'll grow this tensor as we generate new tokens
    generated_ids = input_ids.clone()

    # 2) Generate loop
    for _ in range(max_new_tokens):
        # a) Forward pass to get logits for the last token
        with torch.no_grad():
            outputs = model(generated_ids)
            # outputs shape: [batch=1, seq_len, vocab_size]
            next_token_logits = outputs[0, -1, :]  # last position's logits

        # b) Apply temperature
        next_token_logits = next_token_logits / temperature

        # c) Optionally restrict to top-k
        if top_k > 0:
            # Get top k logits and set the rest to -inf
            values, indices = torch.topk(next_token_logits, top_k)
            mask = torch.full_like(next_token_logits, float('-inf'))
            mask[indices] = values
            next_token_logits = mask

        # d) Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        # e) Append the predicted token
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # 3) Decode to text
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text


#from pretrained local checkpoint, from huggingface, weights
#loading models


if __name__ == "__main__":
    # Load the configuration from the dummy.config file
    config_file = "config/dummy.config"
    config = load_config(config_file)
    # Create the model using the loaded configuration
    model = GPTNeoForCausalLM(config)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    model.apply(lambda m: initialize_weights(m, config))

    x = torch.randint(0, config.vocab_size, (1, 16))  # batch=1, seq_len=16
    logits = model(x)  
    print("Logits shape:", logits.shape)

    # Save the model
    if hasattr(config, 'trained_model_path') and config.trained_model_path is not None:
        torch.save(model.state_dict(), config.trained_model_path + ".pt")
        # Load the model
        loaded_model = GPTNeoForCausalLM(config)
        loaded_model.load_state_dict(torch.load(config.trained_model_path + ".pt"))
        # Compare the state dictionaries using torch.equal
        state_dict_equal = True
        for key in model.state_dict().keys():
            if not torch.equal(model.state_dict()[key], loaded_model.state_dict()[key]):
                state_dict_equal = False
                break
        
        assert state_dict_equal, "Model state dictionaries are not equal"
        print("Model saved and loaded successfully.")

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        prompt = "Hello, my name is"
        output_text = generate_text(model, tokenizer, prompt, max_new_tokens=50,
                                    temperature=0.7, top_k=50)#, device="cuda")
        
        print("Generated:", output_text)


    




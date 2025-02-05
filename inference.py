import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import GPTNeoForCausalLM, load_config, initialize_weights

def remap_keys(hf_state_dict):
    new_state_dict = {}
    for key, value in hf_state_dict.items():
        if "attn.attention" in key:
            new_key = key.replace("attn.attention", "attn")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def load_model_hf(config_file: str, hf_model_name: str) -> GPTNeoForCausalLM:
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_state_dict = hf_model.state_dict()

    config = load_config(config_file)
    local_model = GPTNeoForCausalLM(config)  # Removed initialize_weights apply

    remapped_state_dict = remap_keys(hf_state_dict)

    missing_keys, unexpected_keys = local_model.load_state_dict(remapped_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print("Missing keys in local model:", missing_keys)
        print("Unexpected keys from HF state dict:", unexpected_keys)

    return local_model

def load_model_local(config_file: str, local_checkpoint_path: str) -> GPTNeoForCausalLM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_file)
    local_model = GPTNeoForCausalLM(config)
    
    if local_checkpoint_path is not None:
        # Fixed typo: model -> local_model
        local_model.load_state_dict(torch.load(local_checkpoint_path, map_location=device, weights_only=True))
    else:
        local_model.apply(lambda m: initialize_weights(m, config))

    return local_model

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, indices, values)
                next_token_logits = mask

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    config_file = "config/dummy.config"
    hf_model_name = "roneneldan/TinyStories-1M"
    
    model = load_model_hf(config_file, hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)  # Use hf_model_name for tokenizer
    
    prompt = "Once upon a time, in a small village, there lived a"
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=100)
    print("Generated text (HF):\n", generated_text)

    local_checkpoint_path = "trained_models/tinyLM_1M/epoch28.pt"
    model_local = load_model_local(config_file, local_checkpoint_path)
    generated_text_local = generate_text(model_local, tokenizer, prompt, max_new_tokens=100)
    print("Generated text (Local):\n", generated_text_local)
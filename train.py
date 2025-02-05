# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import wandb
import os
from dotenv import load_dotenv
from pathlib import Path
from model import GPTNeoForCausalLM, initialize_weights, load_config, generate_text
from dataload import TokenizedDataset  # Fixed import
import copy  
import shutil


load_dotenv()

def train(config, use_wandb=False, wandb_project="tiny-stories", wandb_run_name="baseline", checkpoint_path=None):
    # -----------------------------
    # 1. Initialize Weights & Biases
    # -----------------------------
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config.__dict__
        )

    # -----------------------------
    # 2. Training Config
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # -----------------------------
    # 3. Prepare Datasets & DataLoaders
    # -----------------------------
    train_dataset = TokenizedDataset(
        filepath="data/raw/train.txt",
        tokenizer=tokenizer,
        block_size=config.block_size,
        use_data_fraction=config.use_data_fraction,
        split="train",
    )
    val_dataset = TokenizedDataset(
        filepath="data/raw/validation.txt",
        tokenizer=tokenizer,
        block_size=config.block_size,
        use_data_fraction=config.use_data_fraction,
        split="validation",
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------
    # 4. Create Model & Load Checkpoint
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeoForCausalLM(config).to(device)

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        model.apply(lambda m: initialize_weights(m, config))

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    # If continuing from checkpoint, load optimizer state as well (optional)
    if checkpoint_path:
        optimizer_path = checkpoint_path.replace(".pt", "_optimizer.pt")
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer state from: {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    # -----------------------------
    # 5. Training Loop
    # -----------------------------
    global_step = 0
    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait before stopping
    patience_counter = 0  # Counter to track consecutive non-improving epochs
    min_delta = 0.001  # Minimum required improvement to reset patience
    best_model_weights = None  # Track best model weights
    best_model_path = None

    def evaluate():
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(inputs)
                loss = loss_fct(outputs.view(-1, config.vocab_size), labels.view(-1))
                total_loss += loss.item()

        perplexity = torch.exp(torch.clamp(torch.tensor(total_loss / len(val_loader)), max=20))
        return total_loss / len(val_loader), perplexity

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        #total_batches = len(train_loader)
        #half_batch = total_batches // 2  # Calculate halfway point

        for batch_idx, batch in progress_bar:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(inputs)
            loss = loss_fct(outputs.view(-1, config.vocab_size), labels.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if use_wandb:
                wandb.log({"train/loss": loss.item(), "step": global_step})

            # Generate text halfway through the epoch
            #if batch_idx == half_batch - 1:  # Check if current batch is the halfway point
            if global_step % 500 == 0:
                prompt = "Once upon a time"
                generated_text = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    max_new_tokens=50, 
                    temperature=1.0, 
                    top_k=0, 
                    device=device
                )
                print(f"\nGenerated text at epoch {epoch+1}, step {global_step}:\n{generated_text}\n")
                
                if use_wandb:
                    # Log as HTML for better formatting in wandb
                    wandb.log(
                        {"generated_text": wandb.Html(f"<pre>{generated_text}</pre>")}, 
                        step=global_step
                    )
                
                model.train()  # Ensure model returns to training mode after generation

            global_step += 1
        


        # Validation
        val_loss, val_perplexity = evaluate() 
        train_loss = epoch_loss / len(train_loader)

        # Save Checkpoint
        checkpoint_dir = Path("checkpoints") / f"{wandb_project}-{wandb_run_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_file)

        # Save Optimizer State (Optional)
        optimizer_file = checkpoint_dir / f"epoch{epoch+1}_optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_file)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f}")
        if use_wandb:
            wandb.log({
                "loss/train": train_loss,
                "loss/val": val_loss,
                "perplexity/val": val_perplexity,
                "epoch": epoch+1,
                "step": global_step  # Align with training steps
            })
        #early stopping
        improvement = best_val_loss - val_loss
        if val_loss < best_val_loss and improvement > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            best_checkpoint = checkpoint_dir / "best_model.pt"
            torch.save(best_model_weights, best_checkpoint)
            best_model_path = str(best_checkpoint)  # Store the path
            print(f"Saved new best model to {best_checkpoint}")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{patience} (Improvement: {improvement:.6f})")

            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                
                break  # Stop training if patience is exceeded
    
    if use_wandb:
        wandb.finish()

    # Restore best weights if available
    if best_model_weights is not None:
        print("Loading best model weights from checkpoint")
        model.load_state_dict(best_model_weights)
    else:
        print("No best weights found, returning final model")

    return model, best_model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/tiny_lm_1M.config")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tiny-stories")
    parser.add_argument("--wandb_run_name", default="baseline")
    parser.add_argument("--checkpoint_path", default=None, help="Path to model checkpoint to resume training")
    parser.add_argument("--use_collab", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    model,best_model_path=train(config, args.use_wandb, args.wandb_project, args.wandb_run_name, args.checkpoint_path)
    final_path = Path("checkpoints") / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best model saved to {final_path}")

    if args.use_collab:
        project_run_name = str(Path(best_model_path).parent.name)  # Gets "tiny-stories-baseline"
        
        # Create the save path
        drive_save_path = f"/content/drive/MyDrive/UCL_DSML/statnlp/tinystories/{project_run_name}"
        
        # Create directories if they don't exist
        Path(drive_save_path).mkdir(parents=True, exist_ok=True)
        
        # Copy the model
        try:
            shutil.copy2(best_model_path, drive_save_path)
        except:
            print("Failed to copy the model to Google Drive. Please check the paths.")    

        with open('best_model_path.txt', 'w') as f:
            f.write(f"{drive_save_path}/best_model.pt")
        print(f"\nTraining complete. Best model saved to: {drive_save_path}/best_model.pt")

    else:
        final_path = Path("checkpoints") / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        print(f"\nTraining complete. Model saved to {final_path}")

if __name__ == "__main__":
    main()

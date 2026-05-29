# train.py
import os
import glob
import re
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from dataset import SLIDDataset, get_collate_fn
from slid_model import SLIDModel
import config

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, encoding='utf-8', mode='a'), logging.StreamHandler()])

def get_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "slid_model_epoch_*.pt"))
    checkpoints = [c for c in checkpoints if "avg" not in c]
    if not checkpoints: return None, 0
    latest_ckpt, max_epoch = None, 0
    for ckpt in checkpoints:
        match = re.search(r'epoch_(\d+).pt', os.path.basename(ckpt))
        if match and int(match.group(1)) > max_epoch:
            max_epoch = int(match.group(1))
            latest_ckpt = ckpt
    return latest_ckpt, max_epoch

# ==========================================
# Model weight averaging (SWA / checkpoint averaging)
# ==========================================
def average_checkpoints(save_dir, last_k=5):
    checkpoints = glob.glob(os.path.join(save_dir, "slid_model_epoch_*.pt"))
    checkpoints = [c for c in checkpoints if "avg" not in c]
    
    def get_epoch(ckpt_path):
        match = re.search(r'epoch_(\d+).pt', os.path.basename(ckpt_path))
        return int(match.group(1)) if match else -1
        
    checkpoints.sort(key=get_epoch)
    ckpts_to_average = checkpoints[-last_k:]
    
    if not ckpts_to_average:
        logging.warning("No checkpoints found to average.")
        return
        
    logging.info("="*80)
    logging.info(f"[*] Safely averaging the last {len(ckpts_to_average)} checkpoints: {[os.path.basename(c) for c in ckpts_to_average]}")
    
    # Use the last epoch's weights as the base to keep BatchNorm running stats
    base_state_dict = torch.load(ckpts_to_average[-1], map_location='cpu')
    avg_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
    
    # Keep only the parameters that should be averaged (exclude BN buffers)
    keys_to_average = [
        k for k in avg_state_dict.keys() 
        if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked'])
    ]

    for ckpt_path in ckpts_to_average[:-1]:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        for k in keys_to_average:
            avg_state_dict[k] += state_dict[k]
                
    for k in keys_to_average:
        avg_state_dict[k] = torch.div(avg_state_dict[k], float(len(ckpts_to_average)))
        
    avg_save_path = os.path.join(save_dir, "slid_model_epoch_avg.pt")
    torch.save(avg_state_dict, avg_save_path)
    logging.info(f"[*] Successfully saved safe averaged model to {avg_save_path}")
    logging.info("="*80)

# ==========================================
# Generic evaluation routine
# ==========================================
def evaluate_epoch(model, dataloader, criterion, device, desc):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=desc, leave=False, ncols=120)
    
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            with autocast(device_type=device.type):
                logits = model(**inputs)
                loss = criterion(logits, labels)
            
            loss_sum += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss_sum / (step + 1):.4f}", "acc": f"{(correct / total)*100:.2f}%"})
            
    avg_loss = loss_sum / len(dataloader) if len(dataloader) > 0 else 0
    acc = correct / total if total > 0 else 0
    return avg_loss, acc

# ==========================================
# Main training procedure
# ==========================================
def train():
    setup_logging(config.SAVE_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu": config.BATCH_SIZE, config.NUM_WORKERS = 4, 0

    logging.info("="*80)
    logging.info(f"Device: {device} | Model: {config.MODEL_NAME_OR_PATH} | BS: {config.BATCH_SIZE}")

    model_type = "speechbrain" if "speechbrain" in config.MODEL_NAME_OR_PATH.lower() else "huggingface"
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME_OR_PATH) if model_type == "huggingface" else None
    dynamic_collate = get_collate_fn(model_type, feature_extractor)

    # 1. Load the Train / Val / Test splits in turn (by split ID)
    logging.info("Loading Train Dataset (Split 1)...")
    train_dataset = SLIDDataset(config.MANIFEST_PATH, split_id=1, target_sr=config.TARGET_SAMPLE_RATE)
    
    label2id = train_dataset.label2id
    num_classes = len(label2id)
    logging.info(f"Dynamic Num Classes: {num_classes}")

    logging.info("Loading Validation Dataset (Split 2)...")
    val_dataset = SLIDDataset(config.MANIFEST_PATH, split_id=2, target_sr=config.TARGET_SAMPLE_RATE, label2id=label2id)
    
    logging.info("Loading Test Dataset (Split 3)...")
    test_dataset = SLIDDataset(config.MANIFEST_PATH, split_id=3, target_sr=config.TARGET_SAMPLE_RATE, label2id=label2id)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=dynamic_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=dynamic_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=dynamic_collate)

    # 2. Build the model and resume from a checkpoint if available
    model = SLIDModel(config.MODEL_NAME_OR_PATH, num_classes).to(device)
    
    start_epoch = 0
    latest_ckpt, resumed_epoch = get_latest_checkpoint(config.SAVE_MODEL_DIR)
    if latest_ckpt:
        logging.info(f"[*] Resuming from: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        start_epoch = resumed_epoch
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device.type)

    total_steps = len(train_dataloader) * config.EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    if start_epoch > 0:
        past_steps = start_epoch * len(train_dataloader)
        for _ in range(past_steps): scheduler.step()

    # 3. Main training loop
    for epoch in range(start_epoch, config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"--- Starting Epoch {epoch+1}/{config.EPOCHS} | Current LR: {current_lr:.2e} ---")
        
        # ---------- 1. Train ----------
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False, ncols=120)
        
        for step, (inputs, labels) in enumerate(pbar_train):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits = model(labels=labels, **inputs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar_train.set_postfix({
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "loss": f"{train_loss_sum / (step + 1):.4f}", 
                "acc": f"{(train_correct / train_total)*100:.2f}%"
            })
            
        avg_train_loss = train_loss_sum / len(train_dataloader)
        train_acc = train_correct / train_total

        # ---------- 2. Val ----------
        val_loss, val_acc = evaluate_epoch(model, val_dataloader, criterion, device, f"Epoch {epoch+1} [Val]")
        
        # ---------- 3. Test ----------
        test_loss, test_acc = evaluate_epoch(model, test_dataloader, criterion, device, f"Epoch {epoch+1} [Test]")
        
        # ---------- Logging & Save ----------
        logging.info(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train [L: {avg_train_loss:.4f}, A: {train_acc*100:.2f}%] | "
            f"Val [L: {val_loss:.4f}, A: {val_acc*100:.2f}%] | "
            f"Test [L: {test_loss:.4f}, A: {test_acc*100:.2f}%]"
        )
        torch.save(model.state_dict(), os.path.join(config.SAVE_MODEL_DIR, f"slid_model_epoch_{epoch+1}.pt"))

    # After the final epoch, trigger checkpoint averaging
    if epoch == config.EPOCHS - 1:
        logging.info("Training finished. Triggering checkpoint averaging...")
        average_checkpoints(config.SAVE_MODEL_DIR, last_k=min(5, config.EPOCHS))
        
        avg_model_path = os.path.join(config.SAVE_MODEL_DIR, "slid_model_epoch_avg.pt")
        if os.path.exists(avg_model_path):
            logging.info(f"Loading averaged model from {avg_model_path} for final evaluation...")
            model.load_state_dict(torch.load(avg_model_path, map_location=device))
            
            val_loss, val_acc = evaluate_epoch(model, val_dataloader, criterion, device, "Avg Model [Val]")
            test_loss, test_acc = evaluate_epoch(model, test_dataloader, criterion, device, "Avg Model [Test]")
            
            logging.info("="*80)
            logging.info(f"🏆 Final Averaged Model Performance | "
                         f"Val [L: {val_loss:.4f}, A: {val_acc*100:.2f}%] | "
                         f"Test [L: {test_loss:.4f}, A: {test_acc*100:.2f}%]")
            logging.info("="*80)

if __name__ == "__main__":
    train()
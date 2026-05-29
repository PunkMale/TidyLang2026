# task1_identification.py
import os
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from slid_model import SLIDModel
from torch.amp import autocast
from tqdm import tqdm
import config

# ==========================================
# Batched parallel inference dataset
# ==========================================
class EvalAudioDataset(Dataset):
    def __init__(self, wav_paths, target_sr=16000):
        self.wav_paths = wav_paths
        self.target_sr = target_sr

    def __len__(self): return len(self.wav_paths)

    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), path

def get_eval_collate_fn(model_type, feature_extractor):
    def collate_fn(batch):
        waveforms, paths = zip(*batch)
        
        # Truncate to ~10 seconds
        max_len = 160000 
        truncated_wavs = [w[:max_len] if w.shape[0] > max_len else w for w in waveforms]

        if model_type == "huggingface":
            np_wavs = [w.numpy() for w in truncated_wavs]
            inputs = feature_extractor(np_wavs, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
            return {"input_values": inputs["input_values"], "attention_mask": inputs.get("attention_mask")}, paths
        else:
            lengths = torch.tensor([len(w) for w in truncated_wavs])
            padded_wavs = pad_sequence(truncated_wavs, batch_first=True)
            wav_lens = lengths.float() / padded_wavs.shape[1]
            return {"wavs": padded_wavs, "wav_lens": wav_lens}, paths
    return collate_fn

# ==========================================
# Core evaluation logic
# ==========================================
def run_task1_identification(model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # Load the label dictionary built during training
    # ---------------------------------------------------------
    dict_path = config.LABEL_DICT_PATH
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Label dictionary not found at {dict_path}. Did you train the model?")
    label_dicts = torch.load(dict_path)
    id2label = label_dicts['id2label']
    num_classes = len(id2label)
    print(f"[*] Loaded dynamic dictionary with {num_classes} classes.")

    model_type = "speechbrain" if "speechbrain" in config.MODEL_NAME_OR_PATH.lower() else "huggingface"
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME_OR_PATH) if model_type == "huggingface" else None

    print(f"Loading {model_type} model for Task 1...")
    # Pass the dynamic num_classes
    model = SLIDModel(config.MODEL_NAME_OR_PATH, num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Model loaded from {model_weights_path}")
    model.to(device)
    model.eval()

    # 1. Collect absolute paths
    with open(config.TASK1_LID_TXT, 'r', encoding='utf-8') as f:
        wav_filenames = [line.strip() for line in f if line.strip()]
    
    full_wav_paths = [os.path.abspath(os.path.join(config.TASK1_DATA_PATH, name)) for name in wav_filenames]

    # 2. Batched inference
    print("Running Task 1: Fast Language Identification...")
    dataset = EvalAudioDataset(full_wav_paths, target_sr=config.TARGET_SAMPLE_RATE)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
                            shuffle=False, collate_fn=get_eval_collate_fn(model_type, feature_extractor))

    results = []
    
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with autocast(device_type=device.type):
                logits = model(**inputs)
            
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Map predicted ids back to language labels via the dynamic id2label
            for path, pred_id in zip(paths, pred_ids):
                pred_lang = id2label[pred_id]
                results.append(f"{pred_lang}")

    with open(config.TASK1_OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(f"{res}\n")
    print(f"Task 1 results saved to {os.path.abspath(config.TASK1_OUTPUT_TXT)}")

if __name__ == "__main__":
    run_task1_identification(os.path.join(config.SAVE_MODEL_DIR, "slid_model_epoch_avg.pt"))
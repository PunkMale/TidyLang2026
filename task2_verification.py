# task2_verification.py
import os
import torch
import torch.nn.functional as F
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
def run_task2_verification(model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # Load the label dictionary to get num_classes for building the model
    # ---------------------------------------------------------
    dict_path = config.LABEL_DICT_PATH
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Label dictionary not found at {dict_path}.")
    label_dicts = torch.load(dict_path)
    num_classes = len(label_dicts['label2id'])
    print(f"[*] Loaded dynamic dictionary. Architecture requires {num_classes} output nodes.")

    model_type = "speechbrain" if "speechbrain" in config.MODEL_NAME_OR_PATH.lower() else "huggingface"
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME_OR_PATH) if model_type == "huggingface" else None

    print(f"Loading {model_type} model for Task 2...")
    # Pass the dynamic num_classes
    model = SLIDModel(config.MODEL_NAME_OR_PATH, num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Model loaded from {model_weights_path}")
    model.to(device)
    model.eval()

    # 1. Parse task files and collect the unique audio paths
    print("Step 1: Parsing task files to find unique audios...")
    enroll_mapping = {}  # enroll_id -> list of absolute paths
    unique_enroll_wavs = set()
    
    with open(config.TASK2_ENROLL_TSV, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            enroll_id = parts[0]
            paths = [os.path.abspath(os.path.join(config.TASK2_ENROLL_PATH, wav)) for wav in parts[1:]]
            enroll_mapping[enroll_id] = paths
            unique_enroll_wavs.update(paths)

    pairs_list = []      # list of tuples: (enroll_id, absolute_test_path)
    unique_test_wavs = set()
    
    with open(config.TASK2_PAIRS_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            enroll_id = parts[0]
            test_path = os.path.abspath(os.path.join(config.TASK2_DATA_PATH, parts[1]))
            pairs_list.append((enroll_id, test_path))
            unique_test_wavs.update([test_path])

    all_unique_wavs = list(unique_enroll_wavs.union(unique_test_wavs))
    print(f"[*] Found {len(all_unique_wavs)} unique audio files from {len(pairs_list)} test pairs. Deduplication done!")

    # 2. Extract embeddings for all audios in batches
    print("Step 2: Batch Extracting Embeddings...")
    dataset = EvalAudioDataset(all_unique_wavs, target_sr=config.TARGET_SAMPLE_RATE)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
                            shuffle=False, collate_fn=get_eval_collate_fn(model_type, feature_extractor))
    
    emb_dict = {} # path -> embedding tensor (on CPU to save GPU memory)
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="Extracting"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with autocast(device_type=device.type):
                embs = model.extract_embedding(**inputs)
            # Move to CPU immediately to avoid GPU memory buildup / OOM
            embs_cpu = embs.cpu()
            for i, p in enumerate(paths):
                emb_dict[p] = embs_cpu[i]

    # 3. Aggregate and normalize the enrollment profiles
    print("Step 3: Aggregating Enrollment Profiles...")
    enroll_embeddings = {}
    for enroll_id, paths in enroll_mapping.items():
        # Average all embeddings belonging to this enrollment ID
        embs = [emb_dict[p] for p in paths if p in emb_dict]
        if embs:
            mean_emb = torch.stack(embs).mean(dim=0)
            mean_emb = F.normalize(mean_emb, p=2, dim=0) 
            enroll_embeddings[enroll_id] = mean_emb

    # 4. Vectorized batch scoring (cosine similarity)
    print("Step 4: Vectorized Fast Scoring...")
    enroll_tensors = []
    test_tensors = []
    
    for enroll_id, test_path in pairs_list:
        enroll_tensors.append(enroll_embeddings[enroll_id])
        test_tensors.append(emb_dict[test_path])
        
    enroll_tensor = torch.stack(enroll_tensors) # shape: [num_pairs, emb_dim]
    test_tensor = torch.stack(test_tensors)     # shape: [num_pairs, emb_dim]

    # Compute cosine similarity in chunks to bound peak memory (500k per chunk)
    chunk_size = 500000
    all_scores = []
    for i in tqdm(range(0, len(enroll_tensor), chunk_size), desc="Scoring Chunks"):
        s = F.cosine_similarity(enroll_tensor[i:i+chunk_size], test_tensor[i:i+chunk_size], dim=-1)
        all_scores.extend(s.tolist())

    # 5. Write results
    print("Step 5: Writing results to file...")
    with open(config.TASK2_OUTPUT_TXT, 'w', encoding='utf-8') as f:
        # One similarity score per trial, one per line
        for score in all_scores:
            f.write(f"{score:.4f}\n")
            
    print(f"Task 2 complete! Scores saved to {os.path.abspath(config.TASK2_OUTPUT_TXT)}")

if __name__ == "__main__":
    run_task2_verification(os.path.join(config.SAVE_MODEL_DIR, "slid_model_epoch_avg.pt"))
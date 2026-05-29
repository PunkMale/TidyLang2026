# dataset.py
import os
import logging
import random
import math
from pathlib import Path
import torch
import torchaudio
import torchaudio.functional as F_audio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config

class SLIDDataset(Dataset):
    def __init__(self, manifest_path, split_id, target_sr=16000, label2id=None):
        """
        split_id: 1 (Train), 2 (Validation), 3 (Test)
        """
        self.data = []
        self.target_sr = target_sr
        self.split_id = str(split_id)
        self.is_train = (self.split_id == '1')
        
        # --- Data augmentation init ---
        self.use_aug = config.USE_AUGMENTATION if self.is_train else False
        self.aug_prob = config.AUG_PROB
        self.musan_files, self.rir_files = [], []
        
        if self.use_aug:
            if Path(config.MUSAN_ROOT).exists():
                self.musan_files = list(Path(config.MUSAN_ROOT).rglob("*.wav"))
            if Path(config.RIR_ROOT).exists():
                self.rir_files = list(Path(config.RIR_ROOT).rglob("*.wav"))

        # ==========================================
        # Parse the manifest and resolve paths across the two audio roots
        # ==========================================
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        parsed_items = []
        unique_langs = set()

        logging.info(f"Parsing manifest for Split {self.split_id}...")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Support tab- or space-separated fields
                parts = line.split()
                if len(parts) < 3: continue
                
                cur_split, rel_path, lang_label = parts[0], parts[1], parts[2]
                
                # Only resolve paths for items belonging to the requested split
                if cur_split == self.split_id:
                    full_path = None
                    train_cand = os.path.join(config.TRAIN_AUDIO_ROOT, rel_path)
                    val_cand = os.path.join(config.VAL_AUDIO_ROOT, rel_path)

                    # Probe the real physical path
                    if os.path.exists(train_cand):
                        full_path = train_cand
                    elif os.path.exists(val_cand):
                        full_path = val_cand
                        
                    if full_path:
                        parsed_items.append({"path": full_path, "label": lang_label})
                        if self.is_train:
                            unique_langs.add(lang_label)
                    else:
                        logging.warning(f"File missing in both TRAIN and VAL roots: {rel_path}")

        # --- Build or load the label dictionary ---
        if self.is_train and label2id is None:
            logging.info("Building dynamic label dictionary from Training Split...")
            sorted_langs = sorted(list(unique_langs))
            self.label2id = {lang: idx for idx, lang in enumerate(sorted_langs)}
            self.id2label = {idx: lang for idx, lang in enumerate(sorted_langs)}
            
            os.makedirs(config.SAVE_MODEL_DIR, exist_ok=True)
            torch.save({'label2id': self.label2id, 'id2label': self.id2label}, config.LABEL_DICT_PATH)
            logging.info(f"[*] Found {len(self.label2id)} classes! Saved to {config.LABEL_DICT_PATH}")
        else:
            if label2id is None:
                raise ValueError("label2id must be provided for Validation/Test splits!")
            self.label2id = label2id

        # --- Load samples ---
        for item in parsed_items:
            if item["label"] in self.label2id:
                self.data.append(item)

        logging.info(f"Loaded {len(self.data)} valid samples for Split {self.split_id}")

    def __len__(self): return len(self.data)

    def _apply_musan(self, waveform):
        noise_path = random.choice(self.musan_files)
        noise, sr = torchaudio.load(noise_path)
        if sr != self.target_sr: noise = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(noise)
        if noise.shape[0] > 1: noise = noise.mean(dim=0, keepdim=True)
        if noise.shape[1] < waveform.shape[1]: noise = noise.repeat(1, int(math.ceil(waveform.shape[1]/noise.shape[1])))
        start_idx = random.randint(0, noise.shape[1] - waveform.shape[1]) if noise.shape[1] > waveform.shape[1] else 0
        noise = noise[:, start_idx : start_idx + waveform.shape[1]]
        snr = torch.tensor([random.uniform(config.AUG_SNR_MIN, config.AUG_SNR_MAX)])
        if noise.norm(p=2) == 0: return waveform
        return waveform + (waveform.norm(p=2) / noise.norm(p=2) * torch.pow(10, -snr / 20)) * noise

    def _apply_rir(self, waveform):
        rir_path = random.choice(self.rir_files)
        rir, sr = torchaudio.load(rir_path)
        if sr != self.target_sr: rir = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(rir)
        if rir.shape[0] > 1: rir = rir[0:1, :] 
        augmented = F_audio.fftconvolve(waveform, rir / torch.norm(rir, p=2))
        return augmented[:, :waveform.shape[1]]

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform, sr = torchaudio.load(item["path"])
        if sr != self.target_sr: waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)

        if self.use_aug and random.random() < self.aug_prob:
            valid_augs = []
            if self.musan_files: valid_augs.append(self._apply_musan)
            if self.rir_files: valid_augs.append(self._apply_rir)
            if valid_augs: waveform = random.choice(valid_augs)(waveform)

        return waveform.squeeze(0), self.label2id[item["label"]]

def get_collate_fn(model_type, feature_extractor=None):
    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        max_len = 160000 
        truncated_wavs = [w[:max_len] if w.shape[0] > max_len else w for w in waveforms]
        labels = torch.tensor(labels, dtype=torch.long)

        if model_type == "huggingface":
            np_wavs = [w.numpy() for w in truncated_wavs]
            inputs = feature_extractor(np_wavs, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
            return {"input_values": inputs["input_values"], "attention_mask": inputs.get("attention_mask")}, labels
        else:
            lengths = torch.tensor([len(w) for w in truncated_wavs])
            padded_wavs = pad_sequence(truncated_wavs, batch_first=True)
            wav_lens = lengths.float() / padded_wavs.shape[1]
            return {"wavs": padded_wavs, "wav_lens": wav_lens}, labels
    return collate_fn
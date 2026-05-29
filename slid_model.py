# slid_model.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from speechbrain.inference.classifiers import EncoderClassifier
import config

# ==========================================
# Compatibility patch: when loading the VoxLingua107 ECAPA model, SpeechBrain
# tries to download a custom.py that does not exist in the repo, causing a 404.
# This patch returns an empty placeholder file when that file is missing so the
# loading flow completes; it also maps the deprecated use_auth_token to token.
# ==========================================
import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')

    filename = kwargs.get('filename')
    if not filename and len(args) > 1:
        filename = args[1]

    try:
        return _original_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if "404" in str(e).lower() or "not found" in str(e).lower():
            if filename == "custom.py":
                print("[*] custom.py not found in repo; supplying an empty placeholder for SpeechBrain.")
                dummy_path = os.path.abspath("dummy_custom.py")
                if not os.path.exists(dummy_path):
                    with open(dummy_path, "w", encoding="utf-8") as f:
                        f.write("# Empty placeholder required by SpeechBrain loading logic.\n")
                return dummy_path
        raise e

huggingface_hub.hf_hub_download = _patched_hf_hub_download
# ==========================================

# === Classification head option 1: ArcFace (AAM-Softmax) ===
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# === Classification head option 2: RAM-Softmax ===
class RAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, m=0.2, s=30.0, **kwargs):
        super(RAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        # in_features is determined dynamically (adapts to the encoder output dim)
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        print('Initialised RAM-Softmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        
        # Inference mode (no label): return the scaled cosine similarity directly
        if label is None:
            return costh * self.s

        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        if costh_m_s.is_cuda: label_view = label_view.cuda()
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1, costh_m_s.size()[1])

        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        costh_relu = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        
        # Return logits directly; the CrossEntropyLoss in train.py computes the loss
        return costh_relu

# === HuggingFace-only component: Attentive Statistics Pooling ===
class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, input_dim))

    def forward(self, x, mask=None):
        attn_weights = self.attention(x) 
        if mask is not None:
            if mask.shape[1] != x.shape[1]:
                mask = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[1], mode='nearest').squeeze(1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        alpha = F.softmax(attn_weights, dim=1) 
        weighted_mean = torch.sum(alpha * x, dim=1) 
        weighted_var = torch.sum(alpha * (x ** 2), dim=1) - (weighted_mean ** 2)
        weighted_std = torch.sqrt(torch.clamp(weighted_var, min=1e-7)) 
        return torch.cat([weighted_mean, weighted_std], dim=1) 

class SLIDModel(nn.Module):
    def __init__(self, model_name_or_path, num_classes):
        super(SLIDModel, self).__init__()
        self.model_type = "speechbrain" if "speechbrain" in model_name_or_path.lower() else "huggingface"
        print(f"[*] Initializing {self.model_type} backbone...")

        if self.model_type == "speechbrain":
            self.sb_classifier = EncoderClassifier.from_hparams(
                source=model_name_or_path, 
                savedir=f"tmp_{model_name_or_path.replace('/', '_')}",
                run_opts={"device": "cpu"}
            )
            self.compute_features = self.sb_classifier.mods.compute_features
            self.mean_var_norm = self.sb_classifier.mods.mean_var_norm
            self.backbone = self.sb_classifier.mods.embedding_model
            for param in self.backbone.parameters(): param.requires_grad = True
            self.hidden_size = 256 # ECAPA output is fixed at 256
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path)
            self.hidden_size = self.backbone.config.hidden_size
            self.pooling = AttentiveStatisticsPooling(input_dim=self.hidden_size)
            self.hidden_size = self.hidden_size * 2 

        # ==========================================
        # Attach the classification head dynamically based on config.LOSS_TYPE
        # ==========================================
        loss_type = getattr(config, "LOSS_TYPE", "aam").lower()
        if loss_type == "ram":
            print("[*] Using RAM-Softmax classification head.")
            self.head = RAMSoftmax(in_features=self.hidden_size, out_features=num_classes, s=30.0, m=0.2)
        else:
            print("[*] Using ArcFace (AAM-Softmax) classification head.")
            self.head = ArcMarginProduct(in_features=self.hidden_size, out_features=num_classes, s=30.0, m=0.2)

    def extract_embedding(self, **kwargs):
        """Extract utterance-level embeddings (unified for both backbones)."""
        if self.model_type == "speechbrain":
            wavs = kwargs.get("wavs")
            wav_lens = kwargs.get("wav_lens")
            if wav_lens is None: wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
            feats = self.compute_features(wavs)
            feats = self.mean_var_norm(feats, wav_lens)
            embeddings = self.backbone(feats, wav_lens).squeeze(1)
        else:
            input_values = kwargs.get("input_values")
            attention_mask = kwargs.get("attention_mask")
            outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
            embeddings = self.pooling(outputs.last_hidden_state, mask=attention_mask)
            
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, labels=None, **kwargs):
        embeddings = self.extract_embedding(**kwargs)
        # Compute logits through the dynamically selected head
        logits = self.head(embeddings, label=labels)
        return logits
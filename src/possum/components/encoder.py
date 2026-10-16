from ..revisions import resolve
"""Frozen encoder features; a supervised head is fitted separately inside training."""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import model_info


class Encoder:
    def __init__(self, model_id, pooling="mean", device="cpu", revision=None):
        self.model_id = model_id
        self.revision = resolve(model_id, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModel.from_pretrained(model_id, revision=self.revision, attn_implementation="eager").to(device).eval()
        self.pooling, self.device = pooling, device
        self.max_length = 256
        self.model.requires_grad_(False)

    def encode(self, texts, batch_size=8):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                z = hidden[:, 0] if self.pooling == "cls" else (hidden*mask).sum(1)/mask.sum(1)
                out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
        return np.concatenate(out)

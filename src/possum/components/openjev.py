"""Official typed-decision API. Returns log probabilities, not raw model logits."""
import numpy as np
import requests

class OpenJev:
    def __init__(self, endpoint="http://127.0.0.1:3000"):
        self.endpoint = endpoint.rstrip("/")
        self.version = self._get("/v1/version")
    def _get(self, path):
        r = requests.get(self.endpoint+path, timeout=15)
        r.raise_for_status()
        return r.json()
    def score(self, texts):
        rows = []
        for text in texts:
            r = requests.post(self.endpoint+"/v1/systemone", json={
                "model": "openjev", "state": text, "questions": {"ideology": {
                    "type": "choice", "instructions": "Classify the expressed viewpoint in U.S. politics, ignoring author or publisher identity.",
                    "criteria": {"left": "A liberal or left-wing position", "right": "A conservative or right-wing position"}}}}, timeout=180)
            r.raise_for_status()
            probs = r.json()["answers"]["ideology"]["probabilities"]
            p = np.asarray([probs["left"], probs["right"]], dtype=float)
            if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                raise ValueError("Invalid OpenJev probability response")
            rows.append(np.log(np.clip(p/p.sum(), 1e-7, 1)))
        return np.asarray(rows)

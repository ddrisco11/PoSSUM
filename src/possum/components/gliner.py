"""GLiNER full class distribution via the official classification interface."""
import numpy as np

class GLiNER:
    def __init__(self, model_id="fastino/gliner2.5-base-v1"):
        from gliner2.classification import Classifier, ClassificationSchema
        from ..revisions import resolve
        from huggingface_hub import snapshot_download
        self.revision = resolve(model_id)
        self.model_id = model_id
        checkpoint = snapshot_download(model_id, revision=self.revision)
        self.model = Classifier.from_pretrained(checkpoint)
        self.schema = ClassificationSchema().single("political_viewpoint", ["left-wing U.S. political viewpoint", "right-wing U.S. political viewpoint"])
    def score(self, texts):
        rows = []
        labels = ["left-wing U.S. political viewpoint", "right-wing U.S. political viewpoint"]
        for i, text in enumerate(texts):
            result = self.model.classify(text, self.schema).to_dict()
            p = result["political_viewpoint"]["probabilities"]
            v = np.array([p[k] for k in labels], dtype=float)
            if not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
                raise ValueError("Invalid GLiNER distribution")
            rows.append(np.log(np.clip(v/v.sum(), 1e-7, 1)))
            if (i+1) % 200 == 0 or i+1 == len(texts):
                print(f"GLiNER: {i+1}/{len(texts)} passages scored", flush=True)
        return np.asarray(rows)

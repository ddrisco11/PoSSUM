"""GLiNER full class distribution via the official classification interface."""
import numpy as np

class GLiNER:
    def __init__(self, model_id="fastino/gliner2.5-base-v1"):
        from gliner2.classification import Classifier, ClassificationSchema
        self.model = Classifier.from_pretrained(model_id)
        self.schema = ClassificationSchema().single("political_viewpoint", ["left-wing U.S. political viewpoint", "right-wing U.S. political viewpoint"])
    def score(self, texts):
        rows = []
        labels = ["left-wing U.S. political viewpoint", "right-wing U.S. political viewpoint"]
        for text in texts:
            result = self.model.classify(text, self.schema).to_dict()
            p = result["political_viewpoint"]["probabilities"]
            v = np.array([p[k] for k in labels], dtype=float)
            if not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
                raise ValueError("Invalid GLiNER distribution")
            rows.append(np.log(np.clip(v/v.sum(), 1e-7, 1)))
        return np.asarray(rows)

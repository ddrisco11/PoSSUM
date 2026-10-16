from .causal import Causal

class Mistral(Causal):
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3", **kwargs):
        super().__init__(model_id, **kwargs)

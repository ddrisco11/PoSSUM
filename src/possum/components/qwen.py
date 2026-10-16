from .causal import Causal

class Qwen(Causal):
    def __init__(self, model_id="Qwen/Qwen3.5-9B", **kwargs):
        super().__init__(model_id, **kwargs)

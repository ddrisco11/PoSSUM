from .encoder import Encoder

class BGE(Encoder):
    def __init__(self, model_id="BAAI/bge-small-en-v1.5", **kwargs):
        super().__init__(model_id, pooling="cls", **kwargs)
    def encode(self, texts, batch_size=8):
        z = super().encode(texts, batch_size)
        if z.shape[1] != 384:
            raise ValueError("PoSSUM gate requires 384-dimensional BGE embeddings")
        return z

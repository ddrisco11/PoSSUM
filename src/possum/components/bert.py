from .encoder import Encoder

class BERT(Encoder):
    def __init__(self, model_id="google-bert/bert-base-uncased", **kwargs):
        super().__init__(model_id, pooling="mean", **kwargs)

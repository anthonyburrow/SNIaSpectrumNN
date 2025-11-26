from torch import nn, Tensor


class SpectrumModelHead(nn.Module):

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")
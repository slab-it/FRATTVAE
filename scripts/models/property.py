import torch
import torch.nn as nn

class propLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int= 64) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.tensor):
        return self.linear2(self.linear1(z))


class propRank(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)
        self.sm = nn.Softmax(dim= 1)

    def forward(self, z: torch.tensor):
        x = self.linear2(self.linear1(z))
        return self.sm(x)


class PairWiseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
    
    def forward(self, s1: float, s2: float, t: float):
        # t = 1 if x1 > x2
        #     0 if x2 > x1
        #     0.5 otherwise
        o = s1 - s2
        return torch.mean(-t * o + self.sp(o))


PROPMDL = {
    'linear': propLinear,
    'rank': propRank
}
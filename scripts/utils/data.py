import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ListDataset(Dataset):
    def __init__(self, frag_indices: list , positions: list, prop: torch.Tensor) -> None:
        """
        frag_indices, positions: list of torch.Tensors with different lengths
        ecfps, prop: torch.Tensor
        """
        super().__init__()
        self.ecfps = None
        self.frag_indices = frag_indices
        self.positions = positions
        self.prop = prop

    def __len__(self):
        return len(self.frag_indices)
    
    def __getitem__(self, index) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ecfps is None:
            return self.frag_indices[index], self.positions[index], self.prop[index]
        else:
            return self.ecfps[index], self.frag_indices[index], self.positions[index], self.prop[index]

    def set_stereo(self, ecfps):
        self.ecfps = ecfps


def collate_pad_fn(batch):
    frag_indices, positions, props = zip(*batch)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    props = torch.stack(props)

    return frag_indices, positions, props

def collate_stereo_fn(batch):
    ecfps, frag_indices, positions, props = zip(*batch)
    ecfps = torch.stack(ecfps)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    props = torch.stack(props)

    return ecfps, frag_indices, positions, props
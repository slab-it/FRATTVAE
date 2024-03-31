import torch

# Reference from https://github.com/pytorch/tutorials/blob/main/beginner_source/translation_transformer.py
def generate_square_subsequent_mask(length: int, device: torch.device= 'cpu'):
    mask = (torch.triu(torch.ones((length, length), device= device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int= 0, batch_first: bool= True):
    device = src.device
    src_seq_len = src.shape[1] if batch_first else src.shape[0]
    tgt_seq_len = tgt.shape[1] if batch_first else tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device= device).type(torch.bool)

    src_padding_mask = src == pad_idx if batch_first else (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = tgt == pad_idx if batch_first else (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


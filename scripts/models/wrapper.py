import torch
import torch.nn as nn
import torch.nn.functional as F


class PropWrapper(nn.Module):
    def __init__(self, vae, pmodel= None) -> None:
        """
        wrapped VAE-TreeTransformer and property model.
        """
        super().__init__()
        self.vae = vae
        self.pmodel = pmodel

    def forward(self, features: torch.Tensor, positions: torch.Tensor, 
                src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, 
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None):
        
        z, mu, ln_var, output = self.vae(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask, sequential= False)

        if self.pmodel:
            pred = self.pmodel(F.dropout(z, p= self.vae.dropout, training= self.training))
        else:
            pred = None

        return z, mu, ln_var, output, pred
    

class nanEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.pad_idx = padding_idx
        self.emb = nn.Embedding(num_embeddings, embedding_dim= embedding_dim, padding_idx= padding_idx)

    def forward(self, x: torch.Tensor):
        """
        x: shape= (batch_size)
        """
        x = x.nan_to_num(self.pad_idx)
        x = self.emb(x.long())

        return x
    

class nanLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
    
    def forward(self, x: torch.Tensor):
        """
        x: shape= (batch_size)
        """
        nan_mask = x.isnan()
        x = x.nan_to_num(0.0)
        x = self.fc(x.float().unsqueeze(1))
        x[nan_mask] = 0

        return x


class CVAEwrapper(nn.Module):
    def __init__(self, vae: nn.Module, name_conditions: list, num_labels: list) -> None:
        """
        vae: VAE-Transformer
        num_conditions: names of conditions. ex. [MW, QED]
        num_labels: label nums of each conditions. list
        """
        assert len(num_labels) == len(name_conditions)
        super().__init__()
        self.vae = vae
        self.names = name_conditions
        self.cond_embs = nn.ModuleDict()
        for n, l in zip(name_conditions, num_labels):
            if l > 1:   # one-hot label
                self.cond_embs[n] = nanEmbedding(l+1, vae.d_model, padding_idx= l)  # add nan index l
            else:       # continuous value
                self.cond_embs[n] = nanLinear(1, vae.d_model)
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor, conditions: dict,
                src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, 
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None, 
                frag_ecfps: torch.Tensor= None, ndummys: torch.Tensor= None, 
                max_nfrags: int= 20, free_n: bool= False, sequential: bool= None):
        """
        conditions: dict of each condtion labels. ex. {'MW': torch.LongTensor(shape= batch_size,), 'QED': torch.LongTensor}
        """
        conditions = torch.stack([self.cond_embs[key](value) for key, value in conditions.items()], dim= 1)
        z, mu, ln_var, output = self.vae(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask, 
                                         frag_ecfps, ndummys, max_nfrags, free_n, sequential, conditions)
        return z, mu, ln_var, output
    
    def encode(self, features: torch.Tensor, positions: torch.Tensor, conditions: dict,
               src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None):
        conditions = torch.stack([self.cond_embs[key](value) for key, value in conditions.items()], dim= 1)
        z, mu, ln_var = self.vae.encode(features, positions, src_mask, src_pad_mask, conditions)
        return z, mu, ln_var
    
    def decode(self, z: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, conditions: dict,
               tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None):
        conditions = torch.stack([self.cond_embs[key](value) for key, value in conditions.items()], dim= 1)
        output = self.vae.decode(z, features, positions, tgt_mask, tgt_pad_mask, conditions)
        return output
    
    def sequential_decode(self, z: torch.Tensor, conditions: dict,
                          frag_ecfps: torch.Tensor, ndummys: torch.Tensor, 
                          max_nfrags: int= 30, free_n: bool= False, asSmiles: bool= False) -> list:
        conditions = torch.stack([self.cond_embs[key](value) for key, value in conditions.items()], dim= 1)
        outout = self.vae.sequential_decode(z, frag_ecfps, ndummys, max_nfrags, free_n, asSmiles, conditions)
        return outout
    
    def add_condition(self, name_condition: str, num_labels: int):
        self.names.append(name_condition)
        self.cond_embs[name_condition] = nn.Embedding(num_labels, embedding_dim= self.vae.d_model, padding_idx= 0)


class StereoPropWrapper(nn.Module):
    def __init__(self, vae, pmodel= None) -> None:
        """
        wrapped VAE-TreeTransformer and property model.
        """
        super().__init__()
        self.vae = vae
        self.pmodel = pmodel

    def forward(self, ecfps:torch.Tensor, features: torch.Tensor, positions: torch.Tensor, 
                src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, 
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None):
        
        z, mu, ln_var, output, dec_ecfps = self.vae(ecfps, features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask, sequential= False)

        if self.pmodel:
            pred = self.pmodel(F.dropout(z, p= self.vae.dropout, training= self.training))
        else:
            pred = None

        return z, mu, ln_var, output, dec_ecfps, pred
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tree import FragmentTree, get_pad_features
from utils.mask import generate_square_subsequent_mask
from utils.construct import constructMol, isomer_search

from models.fttvae import TreePositionalEncoding

class StereoFTTVAE(nn.Module):
    def __init__(self, num_tokens: int, depth: int, width: int, 
                 ecfp_dim: int= 2048, feat_dim: int= 2048, latent_dim: int= 256,
                 d_model: int= 512, d_ff: int= 2048, num_layers: int= 6, nhead: int= 8, 
                 activation: str= 'gelu', dropout: float= 0.1) -> None:
        super().__init__()
        assert activation in ['relu', 'gelu']
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.ecfp_dim = ecfp_dim
        self.dropout = dropout
        self.depth = depth
        self.width = width

        # Common for encoder and decoder
        self.fc_feat = nn.Sequential(nn.Linear(feat_dim, feat_dim//2),
                                     nn.Linear(feat_dim//2, d_model))
        self.PE = TreePositionalEncoding(d_model= d_model, d_pos= d_model, depth= depth, width= width)

        # transformer encoder
        self.fc_ecfp = nn.Sequential(nn.Linear(ecfp_dim, ecfp_dim//2),
                                     nn.Linear(ecfp_dim//2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff,
                                                   dropout= self.dropout, activation= activation, batch_first= True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        # vae
        self.fc_vae = nn.Sequential(nn.Linear(d_model, latent_dim), 
                                    nn.Linear(latent_dim, 2*latent_dim))
        
        # ecfp decoder
        self.fc_dec_ecfp = nn.Sequential(nn.Linear(latent_dim, ecfp_dim//2),
                                         nn.Tanh(),
                                         nn.Linear(ecfp_dim//2, ecfp_dim))

        # transformer decoder
        self.embed = nn.Embedding(num_embeddings= 1, embedding_dim= d_model)      # <super root>
        self.fc_memory = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff,
                                                   dropout= self.dropout, activation= activation, batch_first= True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers= num_layers)
        self.fc_dec = nn.Linear(d_model, num_tokens)

        # for decode smiles
        self.labels = None


    def forward(self, ecfps: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, 
                src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, 
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None,
                frag_ecfps: torch.Tensor= None, ndummys: torch.Tensor= None, 
                max_nfrags: int= 20, free_n: bool= False, sequential: bool= None):
        """
        encode and decode
        Decode in parallel when training process.
        """
        sequential = not self.training if sequential is None else sequential

        z, mu, ln_var = self.encode(ecfps, features, positions, src_mask, src_pad_mask)
        if sequential:
            output, dec_ecpfs = self.sequential_decode(z, frag_ecfps, ndummys, max_nfrags= max_nfrags, free_n= free_n)
        else:
            output, dec_ecpfs = self.decode(z, features, positions, tgt_mask, tgt_pad_mask)

        return z, mu, ln_var, output, dec_ecpfs


    def encode(self, ecfps: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, 
               src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None):
        """
        features: shape= (Batch_size, Length, feat_dim)
        positions: shape= (Batch_size, Length, depth * width)
        src_mask: source mask for masked attention, shape= (Length+1, Length+1)
        src_pad_mask: shape = (Batch_size, Length+1)
        """
        # positional embbeding
        src = self.fc_feat(features) + self.PE(positions)           # (B, L, d_model),  * math.sqrt(self.d_model)?

        # attach super root
        ecfp_embed = self.fc_ecfp(ecfps).unsqueeze(1)               # (B, 1, d_model)
        src = torch.cat([ecfp_embed, src], dim= 1)                  # (B, L+1, d_model)

        # transformer encoding
        out = self.encoder(src, mask= src_mask, src_key_padding_mask= src_pad_mask)
        out = out[:, 0, :].squeeze(1)
        # out = torch.sum(out, dim= 1).squeeze(1)

        # vae
        mu, ln_var = self.fc_vae(out).chunk(2, dim= -1)
        z = self.reparameterization_trick(mu, ln_var)               # (B, latent_dim)

        return z, mu, ln_var
    

    def decode(self, z: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, 
               tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None):
        """
        z: encoder output. shape= (Batch_size, latent_dim)
        features: shape= (Batch_size, Length, feat_dim)
        positions: shape= (Batch_size, Length, depth * width)
        tgt_mask: target mask for masked attention, shape= (Length+1, Length+1)
        tgt_pad_mask: target mask for padding, shape= (Batch_size, Length+1)

        output: logits of label preditions, shape= (Batch_size, Length+1, num_labels)
        """
        # stereo predition
        dec_ecfps = self.fc_dec_ecfp(z)

        # latent variable to memory
        memory = self.fc_memory(z).unsqueeze(1)                     # (B, 1, d_model)

        # postional embedding 
        tgt = self.fc_feat(features) + self.PE(positions)           # (B, L, d_model)

        # attach supur root
        root_embed = self.embed(tgt.new_zeros(tgt.shape[0], 1).long())    
        tgt = torch.cat([root_embed, tgt], dim= 1)                  # (B, L+1, d_model)

        # transformer decoding
        out = self.decoder(tgt, memory, tgt_mask= tgt_mask, tgt_key_padding_mask= tgt_pad_mask)
        out = self.fc_dec(out)                                      # (B, L+1, num_tokens)

        return out, dec_ecfps
    

    def sequential_decode(self, z: torch.Tensor, frag_ecfps: torch.Tensor, ndummys: torch.Tensor, 
                          max_nfrags: int= 30, free_n: bool= False, asSmiles: bool= False) -> list:
        """
        z: latent variable. shape= (Batch_size, latent_dim)
        frag_ecfps: fragment ecfps. shape= (num_labels, feat_dim)
        ndummys: The degree of a fragment means how many children it has. shape= (num_labels, )
        max_nfrags: the maximum number of fragments
        free_n: if False, tree positional encoding as all nodes have n children.

        output: list of fragment tree
        """
        batch_size = z.shape[0]
        device = z.device

        # latent variabel to memory
        memory = self.fc_memory(z).unsqueeze(1)

        # stereo predition
        dec_ecfps = F.sigmoid(self.fc_dec_ecfp(z))

        # root prediction
        root_embed = self.embed(torch.zeros(batch_size, 1, device= device).long())
        out = self.decoder(root_embed, memory)
        out = self.fc_dec(out)
        root_idxs = out.argmax(dim= -1).flatten()      # (B, )
        
        continues = []
        target_ids = [0] * batch_size
        target_ids_list = [[0] for _ in range(batch_size)]
        tree_list = [FragmentTree() for _ in range(batch_size)]
        for i, idx in enumerate(root_idxs):
            parent_id = tree_list[i].add_node(parent_id= None, feature= frag_ecfps[idx], fid= idx.item(), bondtype= 0)
            assert parent_id == 0
            tree_list[i].set_positional_encoding(parent_id, d_pos= self.depth * self.width)
            continues.append(ndummys[idx].item() > 0)

        nfrags = 1
        while (nfrags < max_nfrags) & (sum(continues) > 0):
            # features
            tgt_mask = generate_square_subsequent_mask(length= nfrags+1).to(device)
            features = get_pad_features(tree_list, key= 'x', max_nodes_num= nfrags).to(device)
            positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= nfrags).to(device)
            assert features.shape[0] == positions.shape[0]

            # forward
            tgt = self.fc_feat(features) + self.PE(positions)
            tgt = torch.cat([root_embed, tgt], dim= 1)              # (B, iter+1, d_model)

            out = self.decoder(tgt, memory, tgt_mask= tgt_mask)
            out = self.fc_dec(out)                                  # (B, iter+1, num_labels)
            
            new_idxs = out[:, -1, :].argmax(dim= -1).flatten()      # (B,)

            # add node
            for i, idx in enumerate(new_idxs):
                if continues[i]:
                    if ndummys[idx] == 0:   # don't generate compounds which have multi fragments.
                        idx = torch.tensor(0)
                    if idx != 0:
                        parent_id = target_ids[i]
                        add_node_id = tree_list[i].add_node(parent_id= parent_id, feature= frag_ecfps[idx], fid= idx.item(), bondtype= 1)
                        parent_fid = tree_list[i].dgl_graph.ndata['fid'][parent_id].item()
                        num_sibling = ndummys[parent_fid].item() - 1 if parent_id > 0 else ndummys[parent_fid].item()
                        if free_n:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling= num_sibling, d_pos= self.depth * self.width)
                        else:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling= self.width, d_pos= self.depth * self.width)
                        level = tree_list[i].dgl_graph.ndata['level'][add_node_id].item()

                        # compare the current number of siblings with the ideal number of siblings
                        if (len(tree_list[i].dgl_graph.predecessors(parent_id)) >= num_sibling):
                            target_ids_list[i].pop(-1)

                        # whether the node has children
                        if (ndummys[idx] > 1) & (self.depth > level):
                            target_ids_list[i].append(add_node_id)

                    continues[i] = bool(target_ids_list[i]) if idx != 0 else False
                    target_ids[i] = target_ids_list[i][-1] if continues[i] else 0
            nfrags += 1

        if asSmiles:
            if self.labels:
                outputs = [constructMol(self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()], tree.adjacency_matrix().tolist()) for tree in tree_list]
            else:
                raise ValueError('If asSmiles= True, please set labels. exaple; self.set_labels(labels)')
            if dec_ecfps is not None:
                outputs = [isomer_search(s, ecfp.numpy()) for s, ecfp in zip(outputs, dec_ecfps)]
        else:
            outputs = tree_list

        return outputs, dec_ecfps


    def reparameterization_trick(self, mu, ln_var):
        eps = torch.randn_like(mu)
        z = mu + torch.exp(ln_var / 2) * eps if self.training else mu

        return z
    
    def set_labels(self, labels):
        if type(labels) == np.ndarray:
            self.labels = labels
        else:
            self.labels = np.array(labels)
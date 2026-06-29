import json
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_
import torch.nn.init as init

class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, input1, input2):
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        return output

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

class ExpertMemory(nn.Module):
    def __init__(self, memory_size, input_dim):
        super().__init__()
        self.key_memory = nn.Parameter(torch.randn(memory_size, input_dim))
        self.value_memory = nn.Parameter(torch.randn(memory_size, input_dim))

    def reset_parameters(self):
        nn.init.orthogonal_(self.key_memory)
        nn.init.orthogonal_(self.value_memory)
        
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)  # [B, N, D]
        key_memory_norm = F.normalize(self.key_memory, dim=-1)  # [M, D]
        value = self.value_memory  # [M, D]

        sim = torch.matmul(x_norm, key_memory_norm.T)  # [B, N, M]
        addressing = F.softmax(sim / 0.1, dim=-1)  # [B, N, M]
        attended = torch.matmul(addressing, value)  # [B, N, D]
        return attended

class DIRL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = cfg.model.transformer_encoder.feat_dim
        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.att_head = cfg.model.transformer_encoder.att_head
        self.embed_dim = cfg.model.transformer_encoder.emb_dim

        self.img = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0),
        )

        self.w_embedding = nn.Embedding(14, int(self.att_dim / 2))
        self.h_embedding = nn.Embedding(14, int(self.att_dim / 2))

        self.mlp = nn.Sequential(
            nn.Linear(self.att_dim, self.att_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.att_dim * 4, self.att_dim)
        )

        self.num_hidden_layers = cfg.model.transformer_encoder.att_layer
        self.transformer = nn.ModuleList([CrossTransformer(self.att_dim, self.att_head)
                                          for i in range(self.num_hidden_layers)])

        self.num_change_types = 6  # 0~4: change types, 5: no change
        self.num_experts_change = 5  # change expert
        self.num_experts_no_change = 1  # no change expert

        # Change experts
        self.expert_modules_change = nn.ModuleList([
            ExpertMemory(memory_size=100, input_dim=self.att_dim * 2)
            for _ in range(self.num_experts_change)
        ])
        
        # No-change expert
        self.expert_module_no_change = ExpertMemory(memory_size=100, input_dim=self.att_dim * 2)

        # change classifier (binary) 0: no change, 1: change
        self.change_classifier = nn.Sequential(
            nn.Conv2d(2*self.att_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 2, kernel_size=3, padding=1)
        )

        # type classifier (5-way)
        self.type_classifier = nn.Sequential(
            nn.Conv2d(2*self.att_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 5, kernel_size=3, padding=1)
        )
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self._reset_parameters()
        self.cls_temperature = self.cfg.cls_temperature
        self.type_temperature = self.cfg.type_temperature

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def _add_positional_embedding(self, input, H, W):
        device = input.device
        pos_w, pos_h = torch.arange(W, device=device), torch.arange(H, device=device)
        embed_w, embed_h = self.w_embedding(pos_w), self.h_embedding(pos_h)
        pos_embed = torch.cat([
            embed_w.unsqueeze(0).repeat(W, 1, 1),
            embed_h.unsqueeze(1).repeat(1, H, 1)
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        return input + pos_embed
      
    def _cdcr_loss(self, input_1, input_2):
        feat_1 = self.mlp(input_1.reshape(-1, self.att_dim))
        feat_2 = self.mlp(input_2.reshape(-1, self.att_dim))
        z_a = (feat_1 - feat_1.mean(0)) / feat_1.std(0)
        z_b = (feat_2 - feat_2.mean(0)) / feat_2.std(0)
        c = z_a.T @ z_b / z_a.size(0)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(self.att_dim - 1, self.att_dim + 1)[:, :-1].pow_(2).sum()
        return on_diag + 0.003 * off_diag

    def compute_classification_losses(self, change_logits_mean, type_logits_mean, change_type_labels, is_change, idx_change):
        losses = {}
        device = change_logits_mean.device
        binary_labels = (change_type_labels != 5).long()  # 1: change, 0: no-change
        change_cls_loss = self.cls_loss_fn(change_logits_mean, binary_labels)
        losses['change_cls_loss'] = change_cls_loss

        if type_logits_mean is not None and idx_change is not None and idx_change.numel() > 0:
            gt_is_change = (change_type_labels[idx_change] != 5)  # [B']
            valid_idx = gt_is_change.nonzero(as_tuple=True)[0]  # [B'_valid]
            if valid_idx.numel() > 0:
                valid_logits = type_logits_mean[valid_idx]               # [B_valid, 5]
                valid_labels = change_type_labels[idx_change[valid_idx]]  # [B_valid]
                type_cls_loss = self.cls_loss_fn(valid_logits, valid_labels)
            else:
                type_cls_loss = torch.tensor(0.0, device=device)
        else:
            type_cls_loss = torch.tensor(0.0, device=device)
        losses['change_type_cls_loss'] = type_cls_loss

        return losses

    def compute_cosine_margin_disentangle_loss(
        self, input_diff, change_type_labels, is_change,
        margin_inter=None, margin_intra=None, lambda_inter=None, lambda_intra=None
    ):
        margin_inter = margin_inter if margin_inter is not None else self.cfg.loss.margin_inter
        margin_intra = margin_intra if margin_intra is not None else self.cfg.loss.margin_intra
        lambda_inter = lambda_inter if lambda_inter is not None else self.cfg.loss.lambda_inter
        lambda_intra = lambda_intra if lambda_intra is not None else self.cfg.loss.lambda_intra

        B, N, D = input_diff.shape
        device = input_diff.device
        input_diff = input_diff.reshape(B * N, D)

        labels_expanded = change_type_labels.unsqueeze(1).expand(B, N).reshape(-1)
        is_change_expanded = is_change.unsqueeze(1).expand(B, N).reshape(-1)

        rep_vectors = []
        for i in range(self.num_change_types):  # 0~4: change, 5: no-change
            if i < 5:
                mask = (labels_expanded == i) & is_change_expanded
            else:
                mask = ~is_change_expanded

            if mask.any():
                feats = input_diff[mask]  # [K, D]
                mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
                rep_vectors.append(mean_feat)
            else:
                rep_vectors.append(None)

        # inter-group
        loss_inter = torch.tensor(0.0, device=device)
        z_n = rep_vectors[5]
        valid_change = 0
        if z_n is not None:
            for i in range(5):
                z_c = rep_vectors[i]
                if z_c is not None:
                    sim = torch.matmul(z_n, z_c.T).squeeze()
                    loss_inter += F.relu(sim - margin_inter) ** 2
                    valid_change += 1
        if valid_change > 0:
            loss_inter /= valid_change

        # intra-group
        loss_intra = torch.tensor(0.0, device=device)
        count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                zi, zj = rep_vectors[i], rep_vectors[j]
                if zi is not None and zj is not None:
                    sim = torch.matmul(zi, zj.T).squeeze()
                    loss_intra += F.relu(sim - margin_intra) ** 2
                    count += 1
        if count > 0:
            loss_intra /= count

        return lambda_inter * loss_inter + lambda_intra * loss_intra

    def compute_cosine_consistency_loss(self, output, output_reverse, expert_number=None):
        output = F.normalize(output, dim=-1)  # [B, N, 2D] before + after
        output_reverse = F.normalize(output_reverse, dim=-1)  # [B, N, 2D] after + before
        
        # [B, N] → [B]
        cos_sim = torch.mean(torch.sum(output * output_reverse, dim=-1), dim=-1)  # per sample
        
        if expert_number in [0, 1, 4, 5]:  # non-directional (0: color, 1: texture, 4: move, 5: no-change)
            loss = 1.0 - cos_sim
        elif expert_number in [2, 3]:  # directional (2: add, 3: drop)
            loss = 1.0 + cos_sim

        return loss.mean()  # average over batch

    def forward(self, input_1, input_2, change_type_labels=None):
        B, C, H, W = input_1.size()
        input_1, input_2 = self.img(input_1), self.img(input_2)  # [B, N, D]
        input_1, input_2 = self._add_positional_embedding(input_1, H, W), self._add_positional_embedding(input_2, H, W)
        input_1, input_2 = input_1.view(B, self.att_dim, -1).permute(0, 2, 1), input_2.view(B, self.att_dim, -1).permute(0, 2, 1)

        cdcr_loss = self._cdcr_loss(input_1, input_2) if self.training else torch.tensor(0.0, device=input_1.device)

        # transformer
        input_1, input_2 = input_1.transpose(0, 1), input_2.transpose(0, 1)
        input_1_pre, input_2_pre = input_1, input_2
        for l in self.transformer:
            input_1, input_2 = l(input_1, input_2), l(input_2, input_1)

        # diff tokens
        input_1_diff = (input_1_pre - input_1).permute(1, 0, 2)
        input_2_diff = (input_2_pre - input_2).permute(1, 0, 2)

        # feature maps
        input_1_map = input_1_diff.transpose(1, 2).reshape(B, self.att_dim, H, W)
        input_2_map = input_2_diff.transpose(1, 2).reshape(B, self.att_dim, H, W)
        paired_feat = torch.cat([input_1_map, input_2_map], dim=1)  # [B, 2D, H, W]

        # change / no-change classification
        change_logits = self.change_classifier(paired_feat)  # [B, 2, H, W]
        change_logits_mean = change_logits.mean(dim=[2, 3])  # [B, 2]
        change_probs = torch.softmax(change_logits_mean / self.cls_temperature, dim=-1)
        is_change = (change_probs.argmax(dim=-1) == 1)  # bool [B]

        # containers
        B, N, D = input_1_diff.shape
        MoE_outputs_bef = torch.zeros(B, N, D, device=input_1.device)
        MoE_outputs_aft = torch.zeros(B, N, D, device=input_1.device)
        consistency_loss = torch.tensor(0.0, device=input_1.device)
        type_logits_mean = None
        idx_change = None
        
        # ---- Change branch ----
        is_actual_change = (change_type_labels != 5)
        use_change = is_change & is_actual_change
        if use_change.any():
            idx_change = use_change.nonzero(as_tuple=True)[0]
            diff_change = paired_feat[idx_change]
            type_logits = self.type_classifier(diff_change)  # [B', 5, H, W]
            type_logits_mean = type_logits.mean(dim=[2, 3])

            change_confidence = change_probs[idx_change, 1].unsqueeze(1).detach()  # [B', 1]
            type_probs = (F.softmax(type_logits_mean / self.type_temperature, dim=-1) * change_confidence).detach() # [B', 5] * [B', 1]

            labels_change = change_type_labels[idx_change]  # [B']
            token_1 = input_1_diff[idx_change]  # [B', N, D]
            token_2 = input_2_diff[idx_change]
            paired_token = torch.cat([token_1, token_2], dim=-1)  # [B', N, 2D]

            if self.training:
                reversed_paired_token = torch.cat([token_2, token_1], dim=-1)  # [B', N, 2D]

            weighted_1 = 0  # [B', N, D]
            weighted_2 = 0
            for i in range(self.num_experts_change):
                weight_i = type_probs[:, i].unsqueeze(-1).unsqueeze(-1)  # [B', 1, 1]
                expert = self.expert_modules_change[i]

                mod = expert(paired_token)  # [B', N, 2D]
                if self.training:
                    mod_rev = expert(reversed_paired_token)  # [B', N, 2D]
                    mask_i = (labels_change == i)  # [B']
                    if mask_i.any():
                        loss_i = self.compute_cosine_consistency_loss(
                            mod[mask_i], mod_rev[mask_i], i
                        )
                        consistency_loss += loss_i

                weighted_1 += weight_i * mod[..., :D]  # [B', N, D]
                weighted_2 += weight_i * mod[..., D:]

            MoE_outputs_bef[idx_change] = weighted_1
            MoE_outputs_aft[idx_change] = weighted_2

        # ---- No-change branch ----
        is_actual_no_change = (change_type_labels == 5)
        use_no_change = (~is_change) & is_actual_no_change
        if use_no_change.any():
            idx_nc = use_no_change.nonzero(as_tuple=True)[0]  # [B''] (no change sample indices)
            token_1_nc = input_1_diff[idx_nc]  # [B'', N, D]
            token_2_nc = input_2_diff[idx_nc]
            paired_token_nc = torch.cat([token_1_nc, token_2_nc], dim=-1)

            if self.training:
                reversed_paired_token_nc = torch.cat([token_2_nc, token_1_nc], dim=-1)  # [B'', N, 2D]

            no_change_confidence = change_probs[idx_nc, 0].unsqueeze(1).unsqueeze(1).detach()  # [B'', 1, 1]
            mod_nc = self.expert_module_no_change(paired_token_nc)  # [B'', N, 2D]
            if self.training:
                mod_rev_nc = self.expert_module_no_change(reversed_paired_token_nc)  # [B'', N, 2D]
                loss_nc = self.compute_cosine_consistency_loss(mod_nc, mod_rev_nc, 5)
                consistency_loss += loss_nc

            weighted_1_nc = mod_nc[..., :D] * no_change_confidence  # [B'', N, D]
            weighted_2_nc = mod_nc[..., D:] * no_change_confidence
            MoE_outputs_bef[idx_nc] = weighted_1_nc
            MoE_outputs_aft[idx_nc] = weighted_2_nc

        # final concat
        final_output = torch.cat([MoE_outputs_bef, MoE_outputs_aft], dim=-1)  # (B, N, 2D)

        if not self.training:
            type_probs_full = torch.zeros(B, 6, device=input_1.device, dtype=input_1.dtype)
            if type_logits_mean is not None and is_change.any():
                type_probs_softmax = F.softmax(type_logits_mean, dim=-1)
                type_probs_full[idx_change, :5] = change_probs[idx_change, 1:2] * type_probs_softmax
            type_probs_full[:, 5] = change_probs[:, 0]
            return input_1_diff, input_2_diff, type_probs_full, final_output

        losses = {}
        if self.training and change_type_labels is not None:
            classification_losses = self.compute_classification_losses(
                change_logits_mean, type_logits_mean, change_type_labels, is_change, idx_change
            )
            losses.update(classification_losses)

        # Disentangle Loss
        losses['expert_disentangle_loss'] = self.compute_cosine_margin_disentangle_loss(final_output, change_type_labels, is_change)

        # Cosine Consistency Loss
        losses['expert_consistency_loss'] = consistency_loss

        return input_1_diff, input_2_diff, cdcr_loss, losses, final_output

class AddSpatialInfo(nn.Module):
    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug

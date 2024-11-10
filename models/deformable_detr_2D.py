# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .utils import inverse_sigmoid
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self,
                d_model=256,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                return_intermediate_dec=False,
                num_feature_levels=4,
                dec_n_points=4, 
                enc_n_points=4,
                two_stage=False,
                rln_attn=False,
                two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, rln_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, query_embed, pos_embeds):
        assert self.two_stage or query_embed is not None

        # 21*512
        # srcs
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])
        # masks
        # torch.Size([2, 8, 8])
        # torch.Size([2, 4, 4])
        # torch.Size([2, 2, 2])
        # torch.Size([2, 1, 1])
        # pos
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        pos_embeds_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # lvl 0
            bs, c, h, w = src.shape  # 2 256 8 8
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)  # 8 8
            src = src.flatten(2).transpose(1, 2)  # 2 64 256
            mask = mask.flatten(1) # 2 64
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # 2 64 256
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # 原来是4*256 所以变成 2*64*256 加上 扩张的 1*1*256
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            pos_embeds_flatten.append(pos_embed)
        pos_embeds_flatten = torch.cat(pos_embeds_flatten, 1)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # pos_embeds_flatten
        # torch.Size([2, 85, 256]) 8*8+4*4+2*2+1*1=85
        # src_flatten
        # torch.Size([2, 85, 256])
        # mask_flatten
        # torch.Size([2, 85])
        # lvl_pos_embed_flatten
        # torch.Size([2, 85, 256])
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # tensor([[8, 8],
        #         [4, 4],
        #         [2, 2],
        #         [1, 1]], device='cuda:0')
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # print(level_start_index)
        # tensor([ 0, 64, 80, 84], device='cuda:0')
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # 输出每个维度（batch=2 n=4）中真正像素的占比 比如第一个 就是第一幅图片在8*8维度中没有任何扩展所以就是1
        # 如果之前是7*5 （HW） 输出就是 5/8 7/8 （WH）
        #
        # tensor([[[1., 1.],
        #          [1., 1.],
        #          [1., 1.],
        #          [1., 1.]],
        #         [[1., 1.],
        #          [1., 1.],
        #          [1., 1.],
        #          [1., 1.]]], device='cuda:0')

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        # 2 85 256

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # # 因为后面做 torch.split，用来划分tensor，可以从数量上划分，还有维度上划分
            # query_embed, tgt = torch.split(query_embed, c, dim=1)
            # 其中c=self.hidden_dim 256
            # query_embed = 前半个 21*256 tgt为后半个 2*256
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # 复制为2 21 256
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # 复制为2 21 256
            reference_points = self.reference_points(query_embed).sigmoid()
            # self.reference_points = nn.Linear(d_model, 2)
            # 2*21*2
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios,  pos_embeds_flatten, query_embed, mask_flatten
        )

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # (8, 8)
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # y 就是高H x就是宽W
            # linspace:
            # 输出等差数列 0.5 8-0.5 分成8分tensor([0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000])
            # torch.meshgrid（）的功能是生成网格，可以用于生成坐标。
            # 函数输入两个数据类型相同的一维张量，
            # 两个输出张量的 行数为第一个输入张量的元素个数，
            #             列数为第二个输入张量的元素个数
            # 其中         第一个输出张量填充第一个输入张量中的元素，各行元素相同；
            #             第二个输出张量填充第二个输入张量中的元素各列元素相同。
            # print(ref_y)
            # tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            #         [1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000],
            #         [2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000, 2.5000],
            #         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
            #         [4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000],
            #         [5.5000, 5.5000, 5.5000, 5.5000, 5.5000, 5.5000, 5.5000, 5.5000],
            #         [6.5000, 6.5000, 6.5000, 6.5000, 6.5000, 6.5000, 6.5000, 6.5000],
            #         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000]])
            # print(ref_x)
            # tensor([[0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000],
            #         [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000]])
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # 将ref_y变成64维度 用[None]添加维度变成1*64
            # 通过[None]将valid_ratios变成2*1*4*2维度
            # 因为H_是高度的大小就是8 valid_ratios是在这一维度中像素没有被pad的占比 本案中为1 没有pad
            # 选择的维度是1 就是(h,w)时候的w * H_
            # 这样得到了 2 * 64的长度
            # print(ref_y)
            # tensor([[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.1875,
            #          0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.3125, 0.3125,
            #          0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.4375, 0.4375, 0.4375,
            #          0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.5625, 0.5625, 0.5625, 0.5625,
            #          0.5625, 0.5625, 0.5625, 0.5625, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875,
            #          0.6875, 0.6875, 0.6875, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125,
            #          0.8125, 0.8125, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375,
            #          0.9375],
            #         [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.1875,
            #          0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.3125, 0.3125,
            #          0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.4375, 0.4375, 0.4375,
            #          0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.5625, 0.5625, 0.5625, 0.5625,
            #          0.5625, 0.5625, 0.5625, 0.5625, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875,
            #          0.6875, 0.6875, 0.6875, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125,
            #          0.8125, 0.8125, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375,
            #          0.9375]])
            # 0.0625=0.5/8  0.9375=7.5/8
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # print(ref_x)
            # tensor([[0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625,
            #          0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875,
            #          0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125,
            #          0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375,
            #          0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625,
            #          0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875,
            #          0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125,
            #          0.9375],
            #         [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625,
            #          0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875,
            #          0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125,
            #          0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375,
            #          0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625,
            #          0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875,
            #          0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125,
            #          0.9375]])
            # print(ref_x.shape)
            # torch.Size([2, 64])
            ref = torch.stack((ref_x, ref_y), -1)
            # # 输入张量信息：
            # # a=[i][j]
            # # b=[i][j]
            #
            # c = stack((a,b), dim=-1)
            #
            # 输出张量信息：
            # 在最后一个维度拼接  跟cat不同 cat将最后一个维度进行合并 维度没有升 这个是加了一个维度
            # c[i][j][0] = a[i][j]
            # c[i][j][1] = b[i][j]

            # print(ref)
            # tensor([[[0.0625, 0.0625],
            #          [0.1875, 0.0625],
            #          [0.3125, 0.0625],
            #          [0.4375, 0.0625],
            #          [0.5625, 0.0625],
            #          [0.6875, 0.0625],
            #          [0.8125, 0.0625],
            #          [0.9375, 0.0625],
            #          [0.0625, 0.1875],
            #          [0.1875, 0.1875],
            #          [0.3125, 0.1875],
            #          [0.4375, 0.1875],
            #          [0.5625, 0.1875],
            #          [0.6875, 0.1875],
            #          [0.8125, 0.1875],
            #          [0.9375, 0.1875],
            #          [0.0625, 0.3125],
            #          [0.1875, 0.3125],
            #          [0.3125, 0.3125],
            #          [0.4375, 0.3125],
            #          [0.5625, 0.3125],
            #          [0.6875, 0.3125],
            #          [0.8125, 0.3125],
            #          [0.9375, 0.3125],
            #          [0.0625, 0.4375],
            #          [0.1875, 0.4375],
            #          [0.3125, 0.4375],
            #          [0.4375, 0.4375],
            #          [0.5625, 0.4375],
            #          [0.6875, 0.4375],
            #          [0.8125, 0.4375],
            #          [0.9375, 0.4375],
            #          [0.0625, 0.5625],
            #          [0.1875, 0.5625],
            #          [0.3125, 0.5625],
            #          [0.4375, 0.5625],
            #          [0.5625, 0.5625],
            #          [0.6875, 0.5625],
            #          [0.8125, 0.5625],
            #          [0.9375, 0.5625],
            #          [0.0625, 0.6875],
            #          [0.1875, 0.6875],
            #          [0.3125, 0.6875],
            #          [0.4375, 0.6875],
            #          [0.5625, 0.6875],
            #          [0.6875, 0.6875],
            #          [0.8125, 0.6875],
            #          [0.9375, 0.6875],
            #          [0.0625, 0.8125],
            #          [0.1875, 0.8125],
            #          [0.3125, 0.8125],
            #          [0.4375, 0.8125],
            #          [0.5625, 0.8125],
            #          [0.6875, 0.8125],
            #          [0.8125, 0.8125],
            #          [0.9375, 0.8125],
            #          [0.0625, 0.9375],
            #          [0.1875, 0.9375],
            #          [0.3125, 0.9375],
            #          [0.4375, 0.9375],
            #          [0.5625, 0.9375],
            #          [0.6875, 0.9375],
            #          [0.8125, 0.9375],
            #          [0.9375, 0.9375]],
            #         [[0.0625, 0.0625],
            #          [0.1875, 0.0625],
            #          [0.3125, 0.0625],
            #          [0.4375, 0.0625],
            #          [0.5625, 0.0625],
            #          [0.6875, 0.0625],
            #          [0.8125, 0.0625],
            #          [0.9375, 0.0625],
            #          [0.0625, 0.1875],
            #          [0.1875, 0.1875],
            #          [0.3125, 0.1875],
            #          [0.4375, 0.1875],
            #          [0.5625, 0.1875],
            #          [0.6875, 0.1875],
            #          [0.8125, 0.1875],
            #          [0.9375, 0.1875],
            #          [0.0625, 0.3125],
            #          [0.1875, 0.3125],
            #          [0.3125, 0.3125],
            #          [0.4375, 0.3125],
            #          [0.5625, 0.3125],
            #          [0.6875, 0.3125],
            #          [0.8125, 0.3125],
            #          [0.9375, 0.3125],
            #          [0.0625, 0.4375],
            #          [0.1875, 0.4375],
            #          [0.3125, 0.4375],
            #          [0.4375, 0.4375],
            #          [0.5625, 0.4375],
            #          [0.6875, 0.4375],
            #          [0.8125, 0.4375],
            #          [0.9375, 0.4375],
            #          [0.0625, 0.5625],
            #          [0.1875, 0.5625],
            #          [0.3125, 0.5625],
            #          [0.4375, 0.5625],
            #          [0.5625, 0.5625],
            #          [0.6875, 0.5625],
            #          [0.8125, 0.5625],
            #          [0.9375, 0.5625],
            #          [0.0625, 0.6875],
            #          [0.1875, 0.6875],
            #          [0.3125, 0.6875],
            #          [0.4375, 0.6875],
            #          [0.5625, 0.6875],
            #          [0.6875, 0.6875],
            #          [0.8125, 0.6875],
            #          [0.9375, 0.6875],
            #          [0.0625, 0.8125],
            #          [0.1875, 0.8125],
            #          [0.3125, 0.8125],
            #          [0.4375, 0.8125],
            #          [0.5625, 0.8125],
            #          [0.6875, 0.8125],
            #          [0.8125, 0.8125],
            #          [0.9375, 0.8125],
            #          [0.0625, 0.9375],
            #          [0.1875, 0.9375],
            #          [0.3125, 0.9375],
            #          [0.4375, 0.9375],
            #          [0.5625, 0.9375],
            #          [0.6875, 0.9375],
            #          [0.8125, 0.9375],
            #          [0.9375, 0.9375]]])
            # print(ref.shape)
            # torch.Size([2, 64, 2])
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # print(reference_points.shape)
        # torch.Size([2, 85, 2])
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 2 85 2 >>> 2 85 1 2    2 4 2 >>> 2 1 4 2 做乘法
        # print(reference_points.shape)
        # torch.Size([2, 85, 4, 2])
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # print(reference_points.shape)
        # torch.Size([2, 85, 4, 2])
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            # 2 85 256

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, rln_attn=False):
        super().__init__()
        self.rln_attn = rln_attn

        # cross attention - avoid sparse attn for rln token
        if self.rln_attn:
            self.cross_attn1 = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout11 = nn.Dropout(dropout)
            self.norm11 = nn.LayerNorm(d_model)
            self.cross_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout12 = nn.Dropout(dropout)
            self.norm12 = nn.LayerNorm(d_model)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
        
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.dropout1 = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, pos, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        if self.rln_attn:
            tgt_obj = tgt[:,:-1,:]
            tgt_rln = tgt[:,-1:,:]
            tgt2 = self.cross_attn1(self.with_pos_embed(tgt_obj, query_pos[:,:-1,:]),
                    reference_points[:,:-1,:],
                    src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt_obj = tgt_obj + self.dropout11(tgt2)
            tgt_obj = self.norm11(tgt_obj)
            tgt2 = self.cross_attn2(query=self.with_pos_embed(tgt_rln, query_pos[:,-1:,:]).transpose(0, 1),
                                   key=self.with_pos_embed(src, pos).transpose(0, 1),
                                   value=src.transpose(0, 1), attn_mask=None,
                                   key_padding_mask=src_padding_mask)[0]
            tgt_rln = tgt_rln + self.dropout12(tgt2.transpose(0, 1))
            tgt_rln = self.norm12(tgt_rln)
            tgt =torch.cat((tgt_obj, tgt_rln),1)
        else:
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                reference_points,
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, pos_embeds,
                query_pos=None, src_padding_mask=None):
        # tgt,  2  21 256                           tgt
        # reference_points, 2 21 2                  reference_points
        # src, 2 85 256                             memory
        # src_spatial_shapes, 4 2     88 44 22 11   spatial_shapes
        # src_level_start_index, 4,   0 64 80 84    level_start_index
        # src_valid_ratios, 2 4 2      111111111    valid_ratios
        # pos_embeds, 2 85 256                      pos_embeds_flatten 88 44 22 11的pos
        # query_pos=None, 2 21 256                  query_embed tgt的前面的部分
        # src_padding_mask=None                     mask_flatten 2 85
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                # print(reference_points[:, :, None].shape)
                # torch.Size([2, 21, 1, 2])
                # print(src_valid_ratios[:, None].shape)
                # torch.Size([2, 1, 4, 2])
                # reference_points_input：2 21 4 2
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, pos_embeds, src_padding_mask)
            # 2 21 256

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:  # None
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate: # false
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:  # false
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
        # 2 21 256     2 21 2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(config):
    return DeformableTransformer(
        d_model=config.MODEL.DECODER.HIDDEN_DIM,
        nhead=config.MODEL.DECODER.NHEADS,
        num_encoder_layers=config.MODEL.DECODER.ENC_LAYERS,
        num_decoder_layers=config.MODEL.DECODER.DEC_LAYERS,
        dim_feedforward=config.MODEL.DECODER.DIM_FEEDFORWARD,
        dropout=config.MODEL.DECODER.DROPOUT,
        activation=config.MODEL.DECODER.ACTIVATION,
        return_intermediate_dec=False,
        num_feature_levels=config.MODEL.DECODER.NUM_FEATURE_LEVELS,
        rln_attn=config.MODEL.DECODER.RLN_ATTN
    )

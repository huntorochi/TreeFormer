# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid
########################################################################################################################


class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.use_mst_train = args.use_mst_train
        self.use_gnn = args.use_gnn

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES

        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)
        
        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM * (2 + config.MODEL.DECODER.RLN_TOKEN),
                                          config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
        else:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*(2+config.MODEL.DECODER.RLN_TOKEN),
                                          config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
            # 因为后面做 torch.split，用来划分tensor，可以从数量上划分，还有维度上划分
            # query_embed, tgt = torch.split(query_embed, c, dim=1)
            # 其中c=self.hidden_dim 256
            # query_embed = 前半个 256 tgt为后半个256
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.encoder.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.encoder.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        # self.decoder.decoder.bbox_embed = None


    def forward(self, samples):
        # 2*1*64*64
        # samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])
        samples = nested_tensor_from_tensor_list(samples)
        # 2*3*64*64  # 不需要变成三倍

        # Deformable Transformer backbone
        features, pos = self.encoder(samples)
        # print(len(features))
        # 3
        # print(len(pos))
        # 3

        # Create 
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # print(src.shape)
            # torch.Size([2, 512, 8, 8])
            # print(mask) >>>就是是不是扩张的像素 由于大家都一样 所以不是
            # tensor([[[False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False]],
            #         [[False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False]]],
            #        device='cuda:0')
            srcs.append(self.input_proj[l](src))
            # self.input_proj = nn.ModuleList([
            #                 nn.Sequential(
            #                     nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),512*256
            #                     nn.GroupNorm(32, self.hidden_dim), 》》》 256
            #                     》》》torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
            #                 )])
            # print(srcs[0].shape)
            # torch.Size([2, 256, 8, 8])

            # 222222222
            # print(src.shape)
            # torch.Size([2, 1024, 4, 4])
            # 。。。
            # mask2*8*8 2*4*4 2*2*2 2*1*1
            # src2*256*8*8 2*256*4*4 2*256*2*2 2*256*1*1
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)  # 3
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    # print(src.shape)
                    # torch.Size([2, 256, 1, 1])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # print(mask)
                # tensor([[[False]],
                #         [[False]]], device='cuda:0') 2*64*64 》》 2*1*1src.shape[-2:]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)
                # print(pos_l.shape)
                # torch.Size([2, 256, 1, 1]) 创造了一个pos_sin
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # 21*512
        # print(srcs[0].shape)
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])
        # print(masks[0].shape)
        # torch.Size([2, 8, 8])
        # torch.Size([2, 4, 4])
        # torch.Size([2, 2, 2])
        # torch.Size([2, 1, 1])
        # print(pos[0].shape)
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])
    
        hs, init_reference, inter_references, _, _ = self.decoder(
            srcs, masks, query_embeds, pos
        )
        # 2 21 256    2 21 2    2 21 2

        object_token = hs[...,:self.obj_token,:]
        # 2 20 256

        class_prob = self.class_embed(object_token)
        # 2 20 2

        coord_loc = self.bbox_embed(object_token).sigmoid()
        # 2 20 4

        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}
        return hs, out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_relationformer(config, args, **kwargs):

    encoder = build_backbone(config)
    decoder = build_deforamble_transformer(config)

    model = RelationFormer(
        encoder,
        decoder,
        config,
        args,
        **kwargs
    )

    return model
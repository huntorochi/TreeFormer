# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
INFTY_COST = 1e+5

def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    nan = np.isnan(cost_matrix).any()
    if nan:
        cost_matrix[np.isnan(cost_matrix)]=INFTY_COST

    return linear_sum_assignment(cost_matrix)

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, config):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_nodes = config.MODEL.MATCHER.C_NODE
        self.cost_class = config.MODEL.MATCHER.C_CLASS

    @torch.no_grad()
    def forward(self, outputs, targets):
        """[summary]

        Args:
            outputs ([type]): [description]
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        # print(outputs.keys())
        # dict_keys(['pred_logits', 'pred_nodes'])
        # print(outputs['pred_logits'].shape)
        # torch.Size([2, 20, 2])
        # print(outputs['pred_nodes'].shape)
        # torch.Size([2, 20, 4])  xywh

        # print(targets.keys())
        # dict_keys(['nodes', 'edges'])
        # print(targets['nodes'])
        # [tensor([[0.7763, 0.0778],
        #         [0.5247, 0.0943],
        #         [1.0000, 0.5104],
        #         [0.0727, 0.0000],
        #         [0.7931, 0.4840],
        #         [0.2609, 0.6906]], device='cuda:0'), tensor([[0.4841, 0.4188],
        #         [0.0000, 0.2914],
        #         [1.0000, 0.5358],
        #         [0.8044, 1.0000],
        #         [0.2271, 1.0000],
        #         [1.0000, 0.1331]], device='cuda:0')]
        # print(targets['edges'])
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        bs, num_queries = outputs['pred_nodes'].shape[:2]
        # 本来是2*20*4 但是取节点坐标没有节点范围wh

        # We flatten to compute the cost matrices in a batch
        out_nodes = outputs['pred_nodes'][...,:2].flatten(0, 1)  # [batch_size * num_queries, 2]
        # 40*2  后面的2 是只取节点坐标没有节点范围wh

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']])
        # print(tgt_nodes)
        # tensor([[0.7763, 0.0778],
        #         [0.5247, 0.0943],
        #         [1.0000, 0.5104],
        #         [0.0727, 0.0000],
        #         [0.7931, 0.4840],
        #         [0.2609, 0.6906],
        #         [0.4841, 0.4188],
        #         [0.0000, 0.2914],
        #         [1.0000, 0.5358],
        #         [0.8044, 1.0000],
        #         [0.2271, 1.0000],
        #         [1.0000, 0.1331]], device='cuda:0') 12*2  第一个有6个 第二个也有6个点

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)
        # 如果 x1 的形状为 B×P×M 并且 x2 的形状为 B×R×M ，则输出的形状为 B×P×R  b是batch m是维度
        # 其中第一个是model的输出为 40*2 第二个是标准的答案 12*2 这个例子B=1
        # 并且使用pred_nodes中的前两个数据也就是节点坐标作为比较 40*2并不是40*4 因为要将节点坐标作为index的判断 不需要维度
        # cost_nodes  40*12 对应的意思是40个预测输出分别与12个答案进行L1计算损失  例：第一行 第一个输出与答案的12个的差距

        # Compute the cls cost
        tgt_ids = torch.cat([torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['nodes']])
        # targets['nodes']是一个list 先看v v的size就是6*2 6*2 两个数组
        # 所以tgt_ids（12,）全是1 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
        cost_class = -outputs["pred_logits"].flatten(0, 1).softmax(-1)[..., tgt_ids]
        # print(outputs["pred_logits"].flatten(0, 1).shape)
        # torch.Size([40, 2])
        # tensor([[ 只看最后的5个
        #         [ 0.6959, -0.2151],
        #         [ 0.6179, -0.6202],
        #         [ 0.9192, -0.4004],
        #         [ 0.5195,  0.8280],
        #         [ 0.1451,  0.3656]], device='cuda:0', requires_grad=True)
        # 并且对最后一个维度（2）进行softmax
        #         [0.7132, 0.2868],
        #         [0.7752, 0.2248],
        #         [0.7891, 0.2109],
        #         [0.4235, 0.5765],
        #         [0.4451, 0.5549]], device='cuda:0')

        # 先做切片
        # 将上面的softmax后的最后一个[..., 1]
        # [..., 1, 1, 1, 1, 1, 1, 1 ,1 ,1, 1, 1, 1]复制tgt_ids份即40*12
        # 然后取反  这里的意思很重要  cost_class是一个预测损失
        # 损失的定义为第二位为1 通过softmax我们可以值到第一位为0 损失为最大 但是通过取反 所以第一位应该是1 pos 第二位0 neg从而变成方向性
        # print(cost_class.shape)
        # torch.Size([40, 12])
        # print(cost_class)
        # tensor([[
        # [-0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868, -0.2868],
        # [-0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248, -0.2248],
        # [-0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109, -0.2109],
        # [-0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765, -0.5765],
        # [-0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549, -0.5549]]

        # Final cost matrix
        C = self.cost_nodes * cost_nodes + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        # 到这里 C变成了 2 20 12 也就是节点的位置  和 预测节点是否为需要的节点的误差矩阵

        sizes = [len(v) for v in targets['nodes']] # [6, 6]
        indices = [linear_sum_assignment_with_inf(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # print(C.split(sizes, -1)[0].shape)
        # torch.Size([2, 20, 6])
        # 这次的例子不是很好 应该是两个节点数不同的图比较好 换成4 8来讲 是先将12个C变成 2 20 4 》2 20 8最后一个维度分离
        # 要计算的是每一次的20个输出对与对应size的最小损失
        # 这个例子里面是先分成2 20 6 和2 20 6
        #   第一组
        #   分别对应对tgt[0]的6个 用第一个batch的20个计算  进行损失计算放在第一个里面(1 20 6)   需要
        #   分别对应对tgt[0]的6个 用第二个batch的20个计算  进行损失计算放在第一个里面(2 20 6)   x
        #   第二组
        #   分别对应对tgt[1]的6个 用第一个batch的20个计算  进行损失计算放在第一个里面(1 20 6)   x
        #   分别对应对tgt[1]的6个 用第二个batch的20个计算  进行损失计算放在第一个里面(2 20 6)   需要
        # 用i来控制得到对应需要的cost
        # [(array([ 7,  8, 12, 13, 18, 19], dtype=int64), array([2, 3, 5, 4, 0, 1], dtype=int64)),
        # (array([ 9, 12, 13, 14, 18, 19], dtype=int64), array([2, 4, 3, 0, 1, 5], dtype=int64))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))] 预测20个中的第九个对应答案6个的第二个


def build_matcher(config):
    return HungarianMatcher(config)

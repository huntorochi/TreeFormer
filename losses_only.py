import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import box_ops_2D
import numpy as np
from torch.nn import MSELoss
import random
from scipy.sparse.csgraph import minimum_spanning_tree
import torch.distributed as dist
from scipy.sparse import find

INFTY_COST = 1e+5
MIN_TEST = 1e-24
MAX_TEST = 1 - 1e-4


# def assignment_with_nan(cost_matrix):
#     cost_matrix = np.asarray(cost_matrix)
#     nan = np.isnan(cost_matrix).any()
#     if nan:
#         cost_matrix[np.isnan(cost_matrix)]=INFTY_COST
#
#     return cost_matrix
#
# def assignment_with_nan_torch(cost_matrix):
#     nan = torch.isnan(cost_matrix).any()
#     if nan:
#         cost_matrix[torch.isnan(cost_matrix)]=INFTY_COST
#
#     return cost_matrix

# function definition
def epsilon_function(epoch, last_epoch, max_epoch, num_epoch, last_num):
    if epoch <= (last_epoch + num_epoch):
        epsilon = max((1 - ((1 - last_num) / num_epoch) * (epoch-last_epoch)), 0)
    else:
        epsilon = last_num
    return epsilon

def epsilon_function_v1(epoch, last_epoch, max_epoch, last_num):
    epsilon = max((1 - ((1 - last_num) / (max_epoch - last_epoch)) * (epoch-last_epoch)), last_num)
    return epsilon

def epsilon_function_v2(epoch, last_epoch, num_epoch, last_num):
    epsilon = max((1 - ((1 - last_num) / num_epoch) * (epoch-last_epoch)), last_num)
    return epsilon

def epsilon_function_v3(epoch, last_epoch, max_epoch, last_num, keep_epoch):
    step = np.floor((epoch - last_epoch) / keep_epoch) * keep_epoch + last_epoch
    epsilon = max((1 - ((1 - last_num) / (max_epoch - last_epoch)) * (step - last_epoch)), last_num)
    return epsilon




# function definition for v5
def epsilon_function_v5(epoch, last_epoch, max_epoch, last_num):
    # calculate the constant k such that epsilon is 0.01 at epoch = max_epoch
    k = (1 / last_num - 1) / (max_epoch - last_epoch)
    epsilon = last_num / (1 + k * (epoch - last_epoch))
    return epsilon

# function definition for v6
def epsilon_function_v6(epoch, last_epoch, max_epoch, last_num, keep_epoch):
    # calculate the constant k such that epsilon is 0.01 at epoch = max_epoch
    k = (1 / last_num - 1) / (max_epoch - last_epoch)
    step = np.floor((epoch - last_epoch) / keep_epoch) * keep_epoch + last_epoch
    if step <= max_epoch:
        epsilon = last_num / (1 + k * (step - last_epoch))
    else:
        epsilon = last_num
    return epsilon

def prims_mst(cost_adj):
    num_of_nodes = cost_adj.shape[0]
    device = cost_adj.device
    postive_inf = 1000
    selected_nodes = torch.zeros((num_of_nodes, 1), device=device).bool()  # 全是false
    mst_adj = torch.zeros_like(cost_adj, device=device)
    while not selected_nodes.all():
        minimum = postive_inf
        start = 0
        end = 0

        for row in range(num_of_nodes):
            if selected_nodes[row]:
                for col in range(num_of_nodes):
                    if (not selected_nodes[col] and cost_adj[row, col] > 0):
                        if cost_adj[row, col] < minimum:
                            minimum = cost_adj[row, col]
                            start, end = row, col

        selected_nodes[end] = True
        mst_adj[start, end] = minimum
        if minimum == postive_inf:
            mst_adj[start, end] = 0
        mst_adj[end, start] = mst_adj[start, end]
    return mst_adj


def random_unit(p):
    assert p >= 0 and p <= 1, "概率P的值应该处在[0,1]之间！"
    if p == 0:  # 概率为0，直接返回False
        return False
    if p == 1:  # 概率为1，直接返回True
        return True
    p_digits = len(str(p).split(".")[1])
    interval_begin = 1
    interval__end = pow(10, p_digits)
    R = random.randint(interval_begin, interval__end)
    if float(R) / interval__end < p:
        return True
    else:
        return False


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, matcher, net, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.net = net
        self.rln_token = config.MODEL.DECODER.RLN_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = {'boxes': config.TRAIN.W_BBOX,
                            'class': config.TRAIN.W_CLASS,
                            'cards': config.TRAIN.W_CARD,
                            'nodes': config.TRAIN.W_NODE,
                            'edges': config.TRAIN.W_EDGE,
                            }
        self.use_mst_train = args.use_mst_train
        self.use_gnn = args.use_gnn

    def loss_class(self, outputs, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # outputs: out['pred_logits']
        # 2 20 2   希望的是最后的2 能正确预测希望输出的番号 其余番号的地方为0
        weight = torch.tensor([0.2, 0.8]).to(outputs.get_device())
        # tensor([0.2000, 0.8000], device='cuda:0')

        idx = self._get_src_permutation_idx(indices)
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # 输出  前六个是第一个0 后6个是1    2组20个预测中对应tgt的番号
        # (tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), tensor([ 7,  8, 12, 13, 18, 19,  9, 12, 13, 14, 18, 19]))

        # targets = torch.zeros(outputs.shape[:-1], dtype=outputs.dtype).to(outputs.get_device())
        # targets[idx] = 1.0

        # targets = targets.unsqueeze(-1)

        # num_nodes = targets.sum()
        # # loss = F.cross_entropy(outputs.permute(0,2,1), targets, weight=weight, reduction='mean')
        # loss = sigmoid_focal_loss(outputs, targets, num_nodes)

        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.get_device())
        # 全是0 2*20 第一列
        targets[idx] = 1.0
        # tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],》》0-7,  0-8, 。。。
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]],》》1-9, 1-12, 。。。
        #        device='cuda:0')
        loss = F.cross_entropy(outputs.permute(0, 2, 1), targets, weight=weight, reduction='mean')
        # {'class': tensor(0.5013, device='cuda:0', grad_fn= < NllLoss2DBackward >)}
        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]
        return loss

    def loss_cardinality(self, outputs, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # outputs: out['pred_logits']
        # 2 20 2   希望的是最后的2 能正确预测希望输出的番号 其余番号的地方为0
        idx = self._get_src_permutation_idx(indices)
        # (tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), tensor([ 7,  8, 12, 13, 18, 19,  9, 12, 13, 14, 18, 19]))
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        # tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]],
        #        device='cuda:0')
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        # tensor([6, 6], device='cuda:0')
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        # print(outputs.argmax(-1))
        # tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]],
        #        device='cuda:0')  >>> ==2-1 也就是其中预测的个数正好为7  7
        # card_pred = (outputs.sigmoid()>0.5).squeeze(-1).sum(1)

        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction='sum') / (
                    outputs.shape[0] * outputs.shape[1])
        # 第一个预测7 tgt6 第二个预测7 tgt6
        # tensor(0.0500, device='cuda:0')
        return loss

    def loss_nodes(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # outputs
        # 2 20 2  只有节点坐标没有范围
        # targets
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
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        num_nodes = sum(len(t) for t in targets)
        # num_nodes 12=6+6

        idx = self._get_src_permutation_idx(indices)
        # 输出  前六个是第一个0 后6个是1    2组20个预测中对应tgt的番号
        # (tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), tensor([ 7,  8, 12, 13, 18, 19,  9, 12, 13, 14, 18, 19]))
        pred_nodes = outputs[idx]
        # tensor([[0.5189, 0.5298],
        #         [0.4993, 0.5018],
        #         [0.4773, 0.5256],
        #         [0.5077, 0.4973],
        #         [0.5011, 0.4751],
        #         [0.5066, 0.4917],
        #         [0.5023, 0.4967],
        #         [0.4905, 0.5183],
        #         [0.5028, 0.4874],
        #         [0.4887, 0.4840],
        #         [0.4934, 0.4847],
        #         [0.5039, 0.4858]], device='cuda:0', grad_fn=<IndexBackward>)
        # 选出对应的节点
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        # tensor([[1.0000, 0.5104],
        #         [0.0727, 0.0000],
        #         [0.2609, 0.6906],
        #         [0.7931, 0.4840],
        #         [0.7763, 0.0778],
        #         [0.5247, 0.0943],
        #         [1.0000, 0.5358],
        #         [0.2271, 1.0000],
        #         [0.8044, 1.0000],
        #         [0.4841, 0.4188],
        #         [0.0000, 0.2914],
        #         [1.0000, 0.1331]], device='cuda:0')
        # 按照
        # [tensor([2, 3, 5, 4, 0, 1])),tensor([2, 4, 3, 0, 1, 5]))]排布target
        loss = F.l1_loss(pred_nodes, target_nodes, reduction='none')  # TODO: check detr for loss function

        loss = loss.sum() / num_nodes

        return loss

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # outputs
        # 2 20 4  拥有节点的范围所以是xywh个
        # targets
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
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        num_boxes = sum(len(t) for t in targets)  # 12
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]
        # tensor([[0.5189, 0.5298, 0.5213, 0.5144],
        #         [0.4993, 0.5018, 0.5084, 0.4955],
        #         [0.4773, 0.5256, 0.4950, 0.5033],
        #         [0.5077, 0.4973, 0.5055, 0.5095],
        #         [0.5011, 0.4751, 0.5042, 0.5009],
        #         [0.5066, 0.4917, 0.5008, 0.4984],
        #         [0.5023, 0.4967, 0.4695, 0.5097],
        #         [0.4905, 0.5183, 0.5066, 0.5013],
        #         [0.5028, 0.4874, 0.4858, 0.5098],
        #         [0.4887, 0.4840, 0.4942, 0.5047],
        #         [0.4934, 0.4847, 0.5194, 0.5009],
        #         [0.5039, 0.4858, 0.5053, 0.4972]], device='cuda:0',
        #        grad_fn=<IndexBackward>)
        # src_boxes = assignment_with_nan_torch(src_boxes)
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 选出坐标xy
        # tensor([[1.0000, 0.5104],
        #         [0.0727, 0.0000],
        #         [0.2609, 0.6906],
        #         [0.7931, 0.4840],
        #         [0.7763, 0.0778],
        #         [0.5247, 0.0943],
        #         [1.0000, 0.5358],
        #         [0.2271, 1.0000],
        #         [0.8044, 1.0000],
        #         [0.4841, 0.4188],
        #         [0.0000, 0.2914],
        #         [1.0000, 0.1331]], device='cuda:0')
        # 这个地方要改一下 最短距离为2.67162269289781以这个大小的正方形才是bbox的大小
        # 两边长为3.778245045839813的矩形 图像h为570 w为190
        width_cat = torch.ones((target_boxes.shape[0], 1), device=target_boxes.device) * (3 / 190)
        height_cat = torch.ones((target_boxes.shape[0], 1), device=target_boxes.device) * (3 / 570)
        target_boxes = torch.cat([target_boxes, width_cat, height_cat], dim=-1)
        # target_boxes = torch.cat([target_boxes, 0.2 * torch.ones(target_boxes.shape, device=target_boxes.device)], dim=-1)
        # 加入wh
        # tensor([[1.0000, 0.5104, 0.2000, 0.2000],
        #         [0.0727, 0.0000, 0.2000, 0.2000],
        #         [0.2609, 0.6906, 0.2000, 0.2000],
        #         [0.7931, 0.4840, 0.2000, 0.2000],
        #         [0.7763, 0.0778, 0.2000, 0.2000],
        #         [0.5247, 0.0943, 0.2000, 0.2000],
        #         [1.0000, 0.5358, 0.2000, 0.2000],
        #         [0.2271, 1.0000, 0.2000, 0.2000],
        #         [0.8044, 1.0000, 0.2000, 0.2000],
        #         [0.4841, 0.4188, 0.2000, 0.2000],
        #         [0.0000, 0.2914, 0.2000, 0.2000],
        #         [1.0000, 0.1331, 0.2000, 0.2000]], device='cuda:0')

        loss = 1 - torch.diag(box_ops_2D.generalized_box_iou(
            box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
            box_ops_2D.box_cxcywh_to_xyxy(target_boxes)))
        # tensor([1.2883, 1.5599, 0.9991, 1.0788, 1.3763, 1.2254, 1.3417, 1.4273, 1.4854,
        #         0.8396, 1.3529, 1.5107], device='cuda:0', grad_fn=<RsubBackward1>)
        loss = loss.sum() / num_boxes  # 1.2905
        return loss

    def loss_edges(self, h, target_nodes, target_edges, indices, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            edge_labels = []
            relation_feature = []
            loss = 0.0
            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge
                # if pos_edge.shape[0] > 300:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:300, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit

                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_labels.append(
                    torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                       rearranged_object_token[all_edges_[:, 1], :],
                                                       relation_token[batch_id, ...].repeat(total_edge, 1)), 1))
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature.append(
                        torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                   rearranged_object_token[all_edges_[:, 1], :]),
                                  1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            #############################################################################
            relation_pred = F.relu(self.net.module.relation_embed(relation_feature))
            # 30, 2
            #####
            nllloss_func = nn.NLLLoss(reduction='mean')
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
            #####
            if relation_pred_softmax_batch.lt(MIN_TEST).any():
                relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST

                relation_pred_softmax_batch = relation_pred_softmax_batch + (
                        relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
            #####
            # relation_pred_softmax_batch = torch.nan_to_num(relation_pred_softmax_batch, nan=INFTY_COST,
            #                                                posinf=INFTY_COST, neginf=-INFTY_COST)
            #####
            relation_pred_softmax_batch = relation_pred_softmax_batch.log()
            nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())
            loss = loss + nlloss_batch
            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')

            # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_infinity(self, h, target_nodes, target_edges, indices, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            edge_labels = []
            relation_feature = []
            loss = 0.0
            nllloss_func = nn.NLLLoss(reduction='mean')
            # loop through each of batch to collect the edge and node

            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 正边的上三角部分
                # mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # inverse_full_adj = full_adj == 0
                # zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # pos_edge = torch.nonzero(zero_upper_triangular)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge
                # if pos_edge.shape[0] > 300:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:300, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit

                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_labels.append(
                    torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]





                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                       rearranged_object_token[all_edges_[:, 1], :],
                                                       relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1))
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature.append(
                        torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                   rearranged_object_token[all_edges_[:, 1], :]),
                                  1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            #############################################################################
            relation_pred = self.net.module.relation_embed(relation_feature)
            # 30, 2
            #####
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
            #####
            if relation_pred_softmax_batch.lt(MIN_TEST).any():
                relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST

                relation_pred_softmax_batch = relation_pred_softmax_batch + (
                        relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
            #####
            # relation_pred_softmax_batch = torch.nan_to_num(relation_pred_softmax_batch, nan=INFTY_COST,
            #                                                posinf=INFTY_COST, neginf=-INFTY_COST)
            #####
            relation_pred_softmax_batch = relation_pred_softmax_batch.log()
            nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())
            loss = loss + nlloss_batch
            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')

            # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]


    def loss_edges_infinity_same_shuffle(self, h, target_nodes, target_edges, indices, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            edge_labels = []
            relation_feature = []
            loss = 0.0
            nllloss_func = nn.NLLLoss(reduction='mean')
            # loop through each of batch to collect the edge and node

            shuffle_pos_lst = []
            shuffle_neg_lst = []

            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # # 正边的上三角部分
                # mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # inverse_full_adj = full_adj == 0
                # zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # pos_edge = torch.nonzero(zero_upper_triangular)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                shuffle_pos_lst.append(shuffle)
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge
                # if pos_edge.shape[0] > 300:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:300, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                shuffle_neg_lst.append(shuffle)
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit

                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_labels.append(
                    torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]





                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                       rearranged_object_token[all_edges_[:, 1], :],
                                                       relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1))
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature.append(
                        torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                   rearranged_object_token[all_edges_[:, 1], :]),
                                  1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            #############################################################################
            relation_pred = self.net.module.relation_embed(relation_feature)
            # 30, 2
            #####
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
            #####
            if relation_pred_softmax_batch.lt(MIN_TEST).any():
                relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST

                relation_pred_softmax_batch = relation_pred_softmax_batch + (
                        relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
            #####
            # relation_pred_softmax_batch = torch.nan_to_num(relation_pred_softmax_batch, nan=INFTY_COST,
            #                                                posinf=INFTY_COST, neginf=-INFTY_COST)
            #####
            relation_pred_softmax_batch = relation_pred_softmax_batch.log()
            nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())
            loss = loss + nlloss_batch
            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')

            # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0], shuffle_pos_lst, shuffle_neg_lst

    def loss_edges_infinity_6(self, h, target_nodes, target_edges, indices, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            edge_labels = []
            relation_feature = []
            loss = 0.0
            nllloss_func = nn.NLLLoss(reduction='mean')
            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])
                # 正边的上三角部分
                # mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # inverse_full_adj = full_adj == 0
                # zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # pos_edge = torch.nonzero(zero_upper_triangular)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]


                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge
                # if pos_edge.shape[0] > 300:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:300, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit

                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_labels.append(
                    torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                       rearranged_object_token[all_edges_[:, 1], :],
                                                       relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1))
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature.append(
                        torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                   rearranged_object_token[all_edges_[:, 1], :]),
                                  1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            #############################################################################
            relation_pred = self.net.module.relation_embed(relation_feature)
            # 30, 2
            #####
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
            #####
            if relation_pred_softmax_batch.lt(MIN_TEST).any():
                relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST

                relation_pred_softmax_batch = relation_pred_softmax_batch + (
                        relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
            #####
            # relation_pred_softmax_batch = torch.nan_to_num(relation_pred_softmax_batch, nan=INFTY_COST,
            #                                                posinf=INFTY_COST, neginf=-INFTY_COST)
            #####
            relation_pred_softmax_batch = relation_pred_softmax_batch.log()
            nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())
            loss = loss + nlloss_batch
            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')

            # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return 6 * loss / h.shape[0]

    def loss_edges_infinity_2(self, h, target_nodes, target_edges, indices, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            edge_labels = []
            relation_feature = []
            loss = 0.0
            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge
                # if pos_edge.shape[0] > 300:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:300, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit

                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_labels.append(
                    torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                       rearranged_object_token[all_edges_[:, 1], :],
                                                       relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1))
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature.append(
                        torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                   rearranged_object_token[all_edges_[:, 1], :]),
                                  1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            #############################################################################
            relation_pred = self.net.module.relation_embed(relation_feature)
            # 30, 2
            #####
            nllloss_func = nn.NLLLoss(reduction='mean')
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1)
            #####
            if relation_pred_softmax_batch.lt(MIN_TEST).any():
                relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST

                relation_pred_softmax_batch = relation_pred_softmax_batch + (
                        relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()
            #####
            # relation_pred_softmax_batch = torch.nan_to_num(relation_pred_softmax_batch, nan=INFTY_COST,
            #                                                posinf=INFTY_COST, neginf=-INFTY_COST)
            #####
            relation_pred_softmax_batch = relation_pred_softmax_batch.log()
            nlloss_batch = nllloss_func(relation_pred_softmax_batch, edge_labels)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())
            loss = loss + nlloss_batch
            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')

            # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return 2 * loss / h.shape[0]

    def loss_edges_mst(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            # [[tensor([0, 1], device='cuda:0'),
            # tensor([1, 3], device='cuda:0'),
            # tensor([2, 4], device='cuda:0'),
            # tensor([4, 5], device='cuda:0')],

            # [tensor([0, 1], device='cuda:0'),
            # tensor([0, 2], device='cuda:0'),
            # tensor([0, 4], device='cuda:0'),
            # tensor([0, 5], device='cuda:0'),
            # tensor([2, 3], device='cuda:0')]]
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # [tensor([[0, 1],
            #         [1, 3],
            #         [2, 4],
            #         [4, 5]], device='cuda:0'), tensor([[0, 1],
            #         [0, 2],
            #         [0, 4],
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # restrict unbalance in the +ve/-ve edge

                # if pos_edge.shape[0] > 20:
                #     # print('Reshaping')
                #     pos_edge = pos_edge[:20, :]

                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()

                    # if dist.get_rank() == 0:
                    #     print(relation_pred_batch)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()
                    #
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.000001
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1).clone()
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch
                    # relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:, 1] - \
                    #                                          max(relation_pred_softmax_batch_true[:, 1])

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)
                    # relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] - \
                    #                                          max(relation_pred_softmax_batch_true[:, 0])

                    # print(relation_pred_softmax_batch_true)
                    #####
                    if relation_pred_softmax_batch.lt(MIN_TEST).any():
                        relation_pred_softmax_batch_true[relation_pred_softmax_batch_true < MIN_TEST] = MIN_TEST

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.000001
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

            # # [print(e,l) for e,l in zip(all_edges, edge_labels)]
            #
            # # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            # relation_feature = torch.cat(relation_feature, 0)
            # # 因为上面两组都是6 6 分别计算  最后得到2组的relation_feature=30*768 30是可以根据边的个数改变的
            # edge_labels = torch.cat(edge_labels, 0).to(h.get_device())
            # # tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            #
            # relation_pred = self.net.relation_embed(relation_feature)
            # # 30, 2
            #
            # # valid_edges = torch.argmax(relation_pred, -1)
            # # print('valid_edge number', valid_edges.sum())
            #
            # loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')
            # # 让relation_pred[edges[i], edge_labels]=最大1其余为接近0
            # # 比如第一组 输出(0.05, 0.95)第五组输出(0.99,0.01)
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_mst_new(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################

                    # # 初始化两个列表
                    # new_pos_edge_list = []
                    # new_neg_edge_list = []
                    #
                    # # 找出每一行的最大值和对应的索引
                    # values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    #
                    # # 根据索引决定哪些元素应该被添加到new_pos_edge和new_neg_list
                    # for i in range(ind.shape[0]):
                    #     if ind[i] == 1:
                    #         new_pos_edge_list.append(all_edges_[i].tolist())
                    #     else:
                    #         new_neg_edge_list.append(all_edges_[i].tolist())
                    #
                    # new_pos_edges = torch.tensor(new_pos_edge_list, device=h.device)
                    # new_neg_edges = torch.tensor(new_neg_edge_list, device=h.device)

                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)

                    # if dist.get_rank() == 0:
                    #     print(relation_pred_batch)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    # 计算 E hat = 1 >>pos MST = 0  removed
                    # E hat = [10, 1000]  changed to [1, 1000] 扩大影响
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 更新new_mst_edge_label_batch
                    # MST = 0

                    epsilon = 0.0
                    # epsilon = epsilon_function_v1(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v2(epoch=epoch, last_epoch=last_epoch, last_num=0.0, num_epoch=100)
                    # epsilon = epsilon_function_v3(epoch=epoch, last_epoch=last_epoch, last_num=0.0,
                    #                               max_epoch=max_epoch, keep_epoch=10)
                    # epsilon = epsilon_function_v5(epoch=epoch, last_epoch=last_epoch,
                    #                               max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v6(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch,
                    #                               last_num=0.0, keep_epoch=10)
                    new_mst_edge_label_batch[mask_pos & ~mask_mst] = epsilon

                    # 计算 E hat = 0 >>neg, MST = 1   added
                    # E hat = [10000, 100]  changed to [10000, 1] 扩大影响
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    # 更新new_mst_not_edge_label_batch
                    new_mst_not_edge_label_batch[mask_neg & mask_mst] = epsilon

                    # 笨方法E hat
                    # for pos in new_pos_edges:
                    #     # E hat = 1
                    #     x, y = pos
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # MST = 0
                    #         if any((all_edges_ == pos).all(dim=1)):
                    #             # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #             position_in_all = torch.where((all_edges_ == pos).all(dim=1))[0]
                    #             # 如果在 all_edges_ 中找到了匹配的位置
                    #             new_mst_edge_label_batch[position_in_all] = 0.0
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             if any((new_neg_edges == test_1).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_1).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0
                    #             if any((new_neg_edges == test_2).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_2).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0

                    # 笨方法GT
                    # for pos_pairs in range(pos_edge.shape[0]):
                    #     # 在这一步中只计算了前面pos的0开始的位置
                    #     x, y = pos_edge[pos_pairs]
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # mst_edge_label_batch[pos_pairs] = \
                    #         #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)
                    #
                    #         mst_edge_label_batch[pos_pairs] = 0.01
                    #         # 对于不是树 但是被选中的要变成0.01 是树不变 1
                    #         new_mst_edge_label_batch[pos_pairs] = 0.0  # mst为0  pos为1
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         # 这里就必须考虑pos+neg才是真的位置
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             # 初始化一个空的列表来保存位置
                    #             if any((neg_edges == test_1).all(dim=1)):
                    #                 # pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_1).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0
                    #             if any((neg_edges == test_2).all(dim=1)):
                    #                 # 如果mst=1的两个对称坐标都不在pos里面  也就是pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_2).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
                    #####
                    # if relation_pred_softmax_batch.lt(MIN_TEST).any():
                    #     relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                    #     relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST
                    #
                    #     relation_pred_softmax_batch = relation_pred_softmax_batch + (
                    #             relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    # mask_MIN_TEST = relation_pred_softmax_batch_true < MIN_TEST
                    # relation_pred_softmax_batch_new = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                    # relation_pred_softmax_batch_new[mask_MIN_TEST] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_mst_new_pos(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # new_mst_not_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    ###########################
                    # new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################

                    # # 初始化两个列表
                    # new_pos_edge_list = []
                    # new_neg_edge_list = []
                    #
                    # # 找出每一行的最大值和对应的索引
                    # values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    #
                    # # 根据索引决定哪些元素应该被添加到new_pos_edge和new_neg_list
                    # for i in range(ind.shape[0]):
                    #     if ind[i] == 1:
                    #         new_pos_edge_list.append(all_edges_[i].tolist())
                    #     else:
                    #         new_neg_edge_list.append(all_edges_[i].tolist())
                    #
                    # new_pos_edges = torch.tensor(new_pos_edge_list, device=h.device)
                    # new_neg_edges = torch.tensor(new_neg_edge_list, device=h.device)

                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)

                    # if dist.get_rank() == 0:
                    #     print(relation_pred_batch)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    # 计算 E hat = 1 >>pos MST = 0  removed
                    # E hat = [10, 1000]  changed to [1, 1000] 扩大影响
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 更新new_mst_edge_label_batch
                    # MST = 0

                    epsilon = 0.0
                    # epsilon = epsilon_function_v1(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v2(epoch=epoch, last_epoch=last_epoch, last_num=0.0, num_epoch=100)
                    # epsilon = epsilon_function_v3(epoch=epoch, last_epoch=last_epoch, last_num=0.0,
                    #                               max_epoch=max_epoch, keep_epoch=10)
                    # epsilon = epsilon_function_v5(epoch=epoch, last_epoch=last_epoch,
                    #                               max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v6(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch,
                    #                               last_num=0.0, keep_epoch=10)
                    new_mst_edge_label_batch[mask_pos & ~mask_mst] = epsilon

                    # 计算 E hat = 0 >>neg, MST = 1   added
                    # E hat = [10000, 100]  changed to [10000, 1] 扩大影响
                    # 计算所有边是否在new_neg_edges中
                    ###########################
                    # mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    # 更新new_mst_not_edge_label_batch
                    ###########################
                    # new_mst_not_edge_label_batch[mask_neg & mask_mst] = epsilon

                    # 笨方法E hat
                    # for pos in new_pos_edges:
                    #     # E hat = 1
                    #     x, y = pos
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # MST = 0
                    #         if any((all_edges_ == pos).all(dim=1)):
                    #             # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #             position_in_all = torch.where((all_edges_ == pos).all(dim=1))[0]
                    #             # 如果在 all_edges_ 中找到了匹配的位置
                    #             new_mst_edge_label_batch[position_in_all] = 0.0
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             if any((new_neg_edges == test_1).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_1).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0
                    #             if any((new_neg_edges == test_2).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_2).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0

                    # 笨方法GT
                    # for pos_pairs in range(pos_edge.shape[0]):
                    #     # 在这一步中只计算了前面pos的0开始的位置
                    #     x, y = pos_edge[pos_pairs]
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # mst_edge_label_batch[pos_pairs] = \
                    #         #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)
                    #
                    #         mst_edge_label_batch[pos_pairs] = 0.01
                    #         # 对于不是树 但是被选中的要变成0.01 是树不变 1
                    #         new_mst_edge_label_batch[pos_pairs] = 0.0  # mst为0  pos为1
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         # 这里就必须考虑pos+neg才是真的位置
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             # 初始化一个空的列表来保存位置
                    #             if any((neg_edges == test_1).all(dim=1)):
                    #                 # pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_1).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0
                    #             if any((neg_edges == test_2).all(dim=1)):
                    #                 # 如果mst=1的两个对称坐标都不在pos里面  也就是pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_2).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
                    #####
                    # if relation_pred_softmax_batch.lt(MIN_TEST).any():
                    #     relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                    #     relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST
                    #
                    #     relation_pred_softmax_batch = relation_pred_softmax_batch + (
                    #             relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    # mask_MIN_TEST = relation_pred_softmax_batch_true < MIN_TEST
                    # relation_pred_softmax_batch_new = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                    # relation_pred_softmax_batch_new[mask_MIN_TEST] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_mst_new_neg(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # new_mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    ###############
                    # new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################

                    # # 初始化两个列表
                    # new_pos_edge_list = []
                    # new_neg_edge_list = []
                    #
                    # # 找出每一行的最大值和对应的索引
                    # values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    #
                    # # 根据索引决定哪些元素应该被添加到new_pos_edge和new_neg_list
                    # for i in range(ind.shape[0]):
                    #     if ind[i] == 1:
                    #         new_pos_edge_list.append(all_edges_[i].tolist())
                    #     else:
                    #         new_neg_edge_list.append(all_edges_[i].tolist())
                    #
                    # new_pos_edges = torch.tensor(new_pos_edge_list, device=h.device)
                    # new_neg_edges = torch.tensor(new_neg_edge_list, device=h.device)

                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)

                    # if dist.get_rank() == 0:
                    #     print(relation_pred_batch)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    # 计算 E hat = 1 >>pos MST = 0  removed
                    # E hat = [10, 1000]  changed to [1, 1000] 扩大影响
                    # 计算所有边是否在new_pos_edges中
                    ##################
                    # mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 更新new_mst_edge_label_batch
                    # MST = 0

                    epsilon = 0.0
                    # epsilon = epsilon_function_v1(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v2(epoch=epoch, last_epoch=last_epoch, last_num=0.0, num_epoch=100)
                    # epsilon = epsilon_function_v3(epoch=epoch, last_epoch=last_epoch, last_num=0.0,
                    #                               max_epoch=max_epoch, keep_epoch=10)
                    # epsilon = epsilon_function_v5(epoch=epoch, last_epoch=last_epoch,
                    #                               max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v6(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch,
                    #                               last_num=0.0, keep_epoch=10)
                    #####################
                    # new_mst_edge_label_batch[mask_pos & ~mask_mst] = epsilon

                    # 计算 E hat = 0 >>neg, MST = 1   added
                    # E hat = [10000, 100]  changed to [10000, 1] 扩大影响
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    # 更新new_mst_not_edge_label_batch
                    new_mst_not_edge_label_batch[mask_neg & mask_mst] = epsilon

                    # 笨方法E hat
                    # for pos in new_pos_edges:
                    #     # E hat = 1
                    #     x, y = pos
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # MST = 0
                    #         if any((all_edges_ == pos).all(dim=1)):
                    #             # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #             position_in_all = torch.where((all_edges_ == pos).all(dim=1))[0]
                    #             # 如果在 all_edges_ 中找到了匹配的位置
                    #             new_mst_edge_label_batch[position_in_all] = 0.0
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             if any((new_neg_edges == test_1).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_1).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0
                    #             if any((new_neg_edges == test_2).all(dim=1)):
                    #                 # E hat = 0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 position_in_all = torch.where((all_edges_ == test_2).all(dim=1))[0]
                    #                 new_mst_not_edge_label_batch[position_in_all] = 0.0

                    # 笨方法GT
                    # for pos_pairs in range(pos_edge.shape[0]):
                    #     # 在这一步中只计算了前面pos的0开始的位置
                    #     x, y = pos_edge[pos_pairs]
                    #     if mst_adj_batch[x, y] == 0:  # 不是树
                    #         # mst_edge_label_batch[pos_pairs] = \
                    #         #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)
                    #
                    #         mst_edge_label_batch[pos_pairs] = 0.01
                    #         # 对于不是树 但是被选中的要变成0.01 是树不变 1
                    #         new_mst_edge_label_batch[pos_pairs] = 0.0  # mst为0  pos为1
                    #
                    # for row in range(mst_adj_batch.shape[0]):
                    #     for col in range(mst_adj_batch.shape[1]):
                    #         # 这里就必须考虑pos+neg才是真的位置
                    #         if mst_adj_batch[row, col] == 1:  # 当mst是1
                    #             test_1 = torch.tensor([row, col]).to(h.device)  # mst=1的对称坐标
                    #             test_2 = torch.tensor([col, row]).to(h.device)
                    #             # 初始化一个空的列表来保存位置
                    #             if any((neg_edges == test_1).all(dim=1)):
                    #                 # pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_1).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0
                    #             if any((neg_edges == test_2).all(dim=1)):
                    #                 # 如果mst=1的两个对称坐标都不在pos里面  也就是pos=0
                    #                 # 使用 torch.where 和 torch.all 来找到匹配的位置
                    #                 p_in_neg = torch.where((neg_edges == test_2).all(dim=1))[0]
                    #                 # 如果在 neg_edges 中找到了匹配的位置
                    #                 if len(p_in_neg) > 0:
                    #                     # 计算在 all_edges_ 中的位置
                    #                     p_in_all = p_in_neg.item() + len(pos_edge)
                    #                     new_mst_not_edge_label_batch[p_in_all] = 0.0

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
                    #####
                    # if relation_pred_softmax_batch.lt(MIN_TEST).any():
                    #     relation_pred_softmax_batch_detach = F.softmax(relation_pred, dim=-1)
                    #     relation_pred_softmax_batch_detach[relation_pred_softmax_batch_detach < MIN_TEST] = MIN_TEST
                    #
                    #     relation_pred_softmax_batch = relation_pred_softmax_batch + (
                    #             relation_pred_softmax_batch_detach - relation_pred_softmax_batch).detach()

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    # mask_MIN_TEST = relation_pred_softmax_batch_true < MIN_TEST
                    # relation_pred_softmax_batch_new = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                    # relation_pred_softmax_batch_new[mask_MIN_TEST] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)
                    # print(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1).shape)
                    # torch.Size([15, 768])
                    # 解释一下 总共所有的边pos+neg=4+11 每条边的features=256 再加上rln token 就等于3*256=768
                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_mst_new_old(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                # mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # # 找出每一行的最大值和对应的索引
                    # values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    # mask_ind = ind == 1
                    # # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    # new_pos_edges = all_edges_[mask_ind]
                    # new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################

                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)

                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    epsilon = 0
                    # epsilon = epsilon_function_v1(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v2(epoch=epoch, last_epoch=last_epoch, last_num=0.0, num_epoch=100)
                    # epsilon = epsilon_function_v3(epoch=epoch, last_epoch=last_epoch, last_num=0.0,
                    #                               max_epoch=max_epoch, keep_epoch=10)
                    # epsilon = epsilon_function_v5(epoch=epoch, last_epoch=last_epoch,
                    #                               max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v6(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch,
                    #                               last_num=0.0, keep_epoch=10)

                    ###########################################################################
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # 这里面的操作是 pos 有   mst 没有   也就是要在mst上下一次生成出来
                    # 所以现在的mst是 1000 10  要继续扩大这个影响  变成 1000 1
                    temp_mask_not_in_mst = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    temp_mask_not_in_mst[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # 现在你可以用 temp_mask_not_in_mst 来更新 new_mst_not_edge_label_batch
                    new_mst_not_edge_label_batch[temp_mask_not_in_mst] = epsilon
                    ###########################################################################
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_edge_label_batch 相同
                    # 这里面的操作是 pos 没有   mst 有   也就是要在mst上下一次去掉
                    # 所以现在的mst是 10 1000  要继续扩大这个影响  变成 1 1000
                    temp_mask_in_mst = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)

                    # 用 mask_in_mst 更新这个临时 mask 的后半部分（对应负边）
                    temp_mask_in_mst[pos_edge.shape[0]:] = (mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 现在你可以用 temp_mask_not_in_mst 来更新 new_mst_edge_label_batch
                    new_mst_edge_label_batch[temp_mask_not_in_mst] = epsilon

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss / h.shape[0]

    def loss_edges_final(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                # tensor([[0, 1],
                #         [1, 3],
                #         [2, 4],
                #         [4, 5]], device='cuda:0')
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                    # tensor([[False, False],
                    #         [False, False],
                    #         [ True, False],
                    #         [False, False]], device='cuda:0')这样一个个改变
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # edge_labels = []
            # relation_feature = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                neg_edges = torch.nonzero(torch.triu(full_adj))
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                # tensor([[0, 1],
                #         [1, 3],
                #         [3, 5],
                #         [1, 2],
                #         [1, 4],
                #         [3, 4],
                #         [0, 2]], device='cuda:0')
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # tensor([[1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')

                # check whether the number of -ve edges are within limit
                # if num_edges - pos_edge.shape[0] < neg_edges.shape[0]:
                #     # 如果pos+neg>40 take neg = 40(最大值)-pos
                #     # total_edge  = 40
                #     take_neg = num_edges - pos_edge.shape[0]
                #     total_edge = num_edges
                # else:
                #     # 不足40条边
                #     take_neg = neg_edges.shape[0]  # 11
                #     total_edge = pos_edge.shape[0] + neg_edges.shape[0]  # 11+4=15

                # if pos_edge.shape[0] < neg_edges.shape[0]:
                #     # neg個數大於pos
                #     take_neg = pos_edge.shape[0]
                #     total_edge = 2 * pos_edge.shape[0]
                # else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2],
                #         [1, 0],
                #         [3, 1],
                #         [2, 4],
                #         [2, 5],
                #         [5, 3],
                #         [2, 1],
                #         [4, 1],
                #         [0, 5],
                #         [0, 4],
                #         [4, 3],
                #         [2, 0]], device='cuda:0')
                # all_edges.append(all_edges_)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                # mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ###########################################################################
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # # 找出每一行的最大值和对应的索引
                    # values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    # mask_ind = ind == 1
                    # # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    # new_pos_edges = all_edges_[mask_ind]
                    # new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################

                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)

                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    epsilon = 0
                    # epsilon = epsilon_function_v1(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v2(epoch=epoch, last_epoch=last_epoch, last_num=0.0, num_epoch=100)
                    # epsilon = epsilon_function_v3(epoch=epoch, last_epoch=last_epoch, last_num=0.0,
                    #                               max_epoch=max_epoch, keep_epoch=10)
                    # epsilon = epsilon_function_v5(epoch=epoch, last_epoch=last_epoch,
                    #                               max_epoch=max_epoch, last_num=0.0)
                    # epsilon = epsilon_function_v6(epoch=epoch, last_epoch=last_epoch, max_epoch=max_epoch,
                    #                               last_num=0.0, keep_epoch=10)

                    ###########################################################################
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # 这里面的操作是 pos 有   mst 没有   也就是要在mst上下一次生成出来  E+
                    # 所以现在的mst是 1000 10  要继续扩大这个影响  变成 1000 1
                    temp_mask_not_in_mst = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    temp_mask_not_in_mst[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # 现在你可以用 temp_mask_not_in_mst 来更新 new_mst_not_edge_label_batch
                    new_mst_not_edge_label_batch[temp_mask_not_in_mst] = epsilon
                    ###########################################################################
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_edge_label_batch 相同
                    # 这里面的操作是 neg 有   mst 有   也就是要在mst上下一次去掉   E-
                    # 所以现在的mst是 10 1000  要继续扩大这个影响  变成 1 1000
                    temp_mask_in_mst = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)

                    # 用 mask_in_mst 更新这个临时 mask 的后半部分（对应负边）
                    temp_mask_in_mst[pos_edge.shape[0]:] = (
                                mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 现在你可以用 temp_mask_not_in_mst 来更新 new_mst_edge_label_batch
                    new_mst_edge_label_batch[temp_mask_in_mst] = epsilon

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    # E+ 要将 fij- 乘以epsilon 对应index=0
                    # E- 要将 fij+ 乘以epsilon 对应index=1
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)
                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                    # relation_pred_softmax_batch[:, 1] = relation_pred_softmax_batch[:, 1] * mst_edge_label_batch
                    # relation_pred_log_softmax_batch = relation_pred_softmax_batch.log()
                    # nlloss_batch = nllloss_func(relation_pred_log_softmax_batch, edge_label_batch)
                    # loss = loss + nlloss_batch

                    # relation_feature.append(relation_feature_batch)

        except Exception as e:
            print(e)
            pdb.set_trace()

        loss_constrained = loss / h.shape[0]
        return loss_unconstrained + loss_constrained

    def loss_edges_final_final(self, h, target_nodes, target_edges, indices, epoch,
                               max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach().cpu()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    epsilon = 0
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # 当E*是pos 但是MST是0  定义为 E*+
                    temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # 当E*是neg 但是MST是1  定义为 E*-
                    temp_mask[pos_edge.shape[0]:] = (
                                mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst) & temp_mask] = epsilon
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst) & temp_mask] = epsilon
                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作


                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        return loss_unconstrained + loss_constrained
    def loss_edges_final_final_GPU(self, h, target_nodes, target_edges, indices, epoch,
                               max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    # with torch.no_grad():
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    epsilon = 0
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # 当E*是pos 但是MST是0  定义为 E*+
                    temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # 当E*是neg 但是MST是1  定义为 E*-
                    temp_mask[pos_edge.shape[0]:] = (
                            mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst) & temp_mask] = epsilon
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst) & temp_mask] = epsilon
                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        return loss_unconstrained + loss_constrained

    def loss_edges_final_final_final_epoch_GPU(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        if last_epoch > 50:

            ######################### constrained ##################################
            """Compute the losses related to the masks: the focal loss and the dice loss.
                       targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
            """
            try:
                # all token except the last one is object token
                object_token = h[..., :self.obj_token, :]
                # 2 20 256

                # last token is relation token
                if self.rln_token > 0:  # 1
                    relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                    # 2 1 256

                # map the ground truth edge indices by the matcher ordering
                target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                                zip(target_edges, indices)]
                # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
                target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                                t in
                                target_edges]
                #         [0, 5],
                #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
                new_target_edges = []  # 按照indices排列tgt_edges
                for t, (_, i) in zip(target_edges, indices):
                    # tensor([2, 3, 5, 4, 0, 1])
                    tx = t.clone().detach()
                    for idx, k in enumerate(i):
                        # idx=0 k=2
                        t[tx == k] = idx
                    new_target_edges.append(t)

                loss = 0.0

                # loop through each of batch to collect the edge and node
                for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                    rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                    full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                    # 6个节点
                    cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                    # cost_adj_batch = torch.zeros_like(full_adj)
                    full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                    full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                    neg_edges = torch.nonzero(torch.triu(full_adj))

                    # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                    mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                    # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                    inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                    # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                    zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                    # 找到值为 0 的元素的坐标
                    pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                    # shuffle edges for undirected edge
                    shuffle = np.random.randn((pos_edge.shape[0])) > 0
                    # [False False False False]  在其中选择大于零的位置
                    to_shuffle = pos_edge[shuffle, :]
                    # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                    pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                    # 将pos_edge shuffle

                    # random sample -ve edge
                    idx_ = torch.randperm(neg_edges.shape[0])
                    # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                    neg_edges = neg_edges[idx_, :].to(h.device)

                    # shuffle edges for undirected edge
                    shuffle = np.random.randn((neg_edges.shape[0])) > 0
                    # [ True  True False False  True  True  True False False  True  True] 11/7

                    to_shuffle = neg_edges[shuffle, :]
                    neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                    take_neg = neg_edges.shape[0]
                    total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                    all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                    edge_label_batch = torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                         torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                    nllloss_func = nn.NLLLoss(reduction='none')
                    mst_edge_label_batch = torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                         torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                    # 这里面用的全是1 因为之后要乘以softmax的第二列
                    # edge_labels.append(edge_label_batch)
                    # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                    # print(epoch, last_epoch, max_epoch)

                    new_mst_edge_label_batch = torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                         torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                    new_mst_not_edge_label_batch = torch.cat(
                        (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                         torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                    # concatenate object token pairs with relation token
                    if self.rln_token > 0:
                        relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                            rearranged_object_token[all_edges_[:, 1], :],
                                                            relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                        relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                        ####################################################################################################
                        # 在假的中取得成果
                        relation_pred_batch_fake = F.relu(relation_pred_batch).detach()
                        # 得到 e hat
                        ###########################################################################
                        # 首先计算y_hat ij
                        # 所需要的是E hat  和 E（MST）
                        # 其中 E 是由new pos   new neg组成
                        # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                        # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                        # 找出每一行的最大值和对应的索引
                        values, ind = torch.max(relation_pred_batch_fake, dim=1)
                        # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                        mask_ind = ind == 1
                        # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                        new_pos_edges = all_edges_[mask_ind]
                        new_neg_edges = all_edges_[~mask_ind]
                        ###########################################################################
                        relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                        cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                        x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                        cost_adj_batch[x, y] = cost_pred_batch
                        cost_adj_batch[y, x] = cost_pred_batch
                        cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                        # for num_pairs in range(all_edges_.shape[0]):
                        #     x, y = all_edges_[num_pairs]
                        #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                        # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                        mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                        mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                        # 将numpy数组转换为torch tensor，并移动到GPU上
                        mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                        epsilon = 0
                        # 计算 E=1 >>new pos MST = 0  这个就是E-
                        # 计算所有边是否在new_pos_edges中
                        mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                        # 计算所有边是否在mst_adj_batch中
                        mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                        # 计算 E=0 >>new neg MST = 1  这个就是E+
                        # 计算所有边是否在new_neg_edges中
                        mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                        ###########################################################################
                        # 现在计算E*+ E*-
                        # 所需要的是E* 真值  和 E（MST）
                        # 其中 E* 是由pos   neg组成
                        # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                        # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                        # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                        # 进行制约  之外的不进行
                        # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                        temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                        # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                        # 当E*是pos 但是MST是0  定义为 E*+
                        temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                        # 当E*是neg 但是MST是1  定义为 E*-
                        temp_mask[pos_edge.shape[0]:] = (
                                mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                        # 更新new_mst_edge_label_batch
                        # 创建布尔类型条件
                        # 计算 E=1 >>new pos MST = 0  这个就是E-
                        new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = epsilon
                        # 更新new_mst_not_edge_label_batch
                        # 计算 E=0 >>new neg MST = 1  这个就是E+
                        new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = epsilon


                        ###########################################################################
                        relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                        relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                        relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                        relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                        ##### 控制界限
                        if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                            mask = relation_pred_softmax_batch_true < MIN_TEST
                            relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                            relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                        #####
                        relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                        nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                        # 只包括 E*+ 和 E*- 的损失
                        # loss 的时候只对temp_mask的地方进行loss的更新
                        loss_mask = temp_mask.to(dtype=nlloss_batch.dtype, device=nlloss_batch.device).detach()
                        nlloss = (nlloss_batch * loss_mask).sum() / max(loss_mask.sum(), 1)
                        loss = loss + nlloss

                    else:
                        relation_feature_batch = torch.cat(
                            (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                        relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                        # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                        # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                        relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                        ###########################################################################
                        cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                        for num_pairs in range(all_edges_.shape[0]):
                            x, y = all_edges_[num_pairs]
                            cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                            cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                        cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                        # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                        # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                        mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                        mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                        # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                        for pos_pairs in range(pos_edge.shape[0]):
                            x, y = pos_edge[pos_pairs]
                            if mst_adj_batch[x, y] == 0:  # 不是树
                                # mst_edge_label_batch[pos_pairs] = \
                                #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                                mst_edge_label_batch[pos_pairs] = 0.01
                                # 对于不是树 但是被选中的要变成0.01 是树不变 1

                        ###########################################################################
                        relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                        relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                                 1] * mst_edge_label_batch

                        relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                                 relation_pred_softmax_batch_true[:, 1] * (
                                                                         1 - mst_edge_label_batch)

                        relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                        nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                        loss = loss + nlloss_batch


            except Exception as e:
                print(e)
                pdb.set_trace()
            loss_constrained = loss / h.shape[0]
            return loss_unconstrained + loss_constrained
        else:
            return loss_unconstrained

    def loss_edges_final_final_final_new_GPU(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
                   这里让E*+之类的与预测值做交集  不是mst
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='none')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    epsilon = 0
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # 当E*是pos 但是MST是0  定义为 E*+
                    temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # 当E*是neg 但是MST是1  定义为 E*-
                    temp_mask[pos_edge.shape[0]:] = (
                            mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 如果真值和预测值相同，则需要后来计算epsilon
                    # 其实这里与参考MST的原理是一致的  当E+的时候，说明Ehat是-  E*-说明真值是-

                    # 也就是说 当gt和ehat相同 但是mst不同的时候需要加入mask
                    # 当gt和ehat不同的时候，不加入mask 因为需要unconst来计算就好
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # temp_mask[:pos_edge.shape[0]] = (pos_edge.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # temp_mask[pos_edge.shape[0]:] = (neg_edges.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)


                    #
                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = epsilon
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = epsilon

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    nlloss_batch = nlloss_batch.clone()
                    # temp_mask 1的地方是E*+ E*-  需要保留，所以，其余的地方要变成0
                    nlloss_batch[~temp_mask] = 0

                    # loss_mask = temp_mask.to(dtype=nlloss_batch.dtype, device=nlloss_batch.device).detach()
                    nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return loss_unconstrained + loss_constrained

    def loss_edges_final_final_final1_GPU(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                new_mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = F.relu(relation_pred_batch).detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    epsilon = 0
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # # 当E*是pos 但是MST是0  定义为 E*+
                    # temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # # 当E*是neg 但是MST是1  定义为 E*-
                    # temp_mask[pos_edge.shape[0]:] = (
                    #         mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = epsilon
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = epsilon

                    ###########################################################################
                    relation_pred_batch_true = F.relu(relation_pred_batch).clone()
                    relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return loss_unconstrained + 5 * loss_constrained

    def loss_edges_final_final_final1_GPU_add_infinity(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges_infinity(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # new_mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                #
                # new_mst_not_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    mst_adj_batch[mst_adj_batch != 0] = 1

                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # # 当E*是pos 但是MST是0  定义为 E*+
                    # temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # # 当E*是neg 但是MST是1  定义为 E*-
                    # temp_mask[pos_edge.shape[0]:] = (
                    #         mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = 1
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = 1

                    new_mst_edge_label_batch_mask = new_mst_edge_label_batch.to(torch.bool)
                    new_mst_not_edge_label_batch_mask = new_mst_not_edge_label_batch.to(torch.bool)

                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] -= 100
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] -= 100

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 1 * loss_unconstrained + 5 * loss_constrained


    def loss_edges_final_final_final1_GPU_change_infinity(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, epsilon, num_edges=500):
        loss_unconstrained = self.loss_edges_infinity(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0
            # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
            nllloss_func = nn.NLLLoss(reduction='mean')

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)


                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # new_mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                #
                # new_mst_not_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)
                    mst_adj_batch[mst_adj_batch != 0] = 1

                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # # 当E*是pos 但是MST是0  定义为 E*+
                    # temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # # 当E*是neg 但是MST是1  定义为 E*-
                    # temp_mask[pos_edge.shape[0]:] = (
                    #         mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = 1
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = 1

                    new_mst_edge_label_batch_mask = new_mst_edge_label_batch.to(torch.bool)
                    new_mst_not_edge_label_batch_mask = new_mst_not_edge_label_batch.to(torch.bool)

                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -epsilon
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -epsilon

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 3 * loss_unconstrained + 3 * loss_constrained


    def loss_edges_final_final_final1_GPU_change_infinity_less_mem_cpu(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, epsilon, num_edges=500):
        loss_unconstrained = self.loss_edges_infinity(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                # tx = t.clone().detach()
                tx = t.detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0
            # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
            nllloss_func = nn.NLLLoss(reduction='mean')

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                # inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                inverse_full_adj = full_adj == 0
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :]

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # take_neg = neg_edges.shape[0]
                # Take the desired number of negative edges, reducing the total edges
                # take_neg = min(neg_edges.shape[0], pos_edge.shape[0] * 2)  # Negative edges are twice the positive edges
                # total_edge = pos_edge.shape[0] + take_neg

                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0).to(h.device)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)


                # mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # new_mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                #
                # new_mst_not_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach().cpu()
                    all_edges_ = all_edges_.cpu()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    # cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)
                    cost_adj_batch.triu_(diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = torch.tensor(mst_adj_batch)
                    mst_adj_batch[mst_adj_batch != 0] = 1

                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # # 当E*是pos 但是MST是0  定义为 E*+
                    # temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # # 当E*是neg 但是MST是1  定义为 E*-
                    # temp_mask[pos_edge.shape[0]:] = (
                    #         mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = 1
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = 1

                    new_mst_edge_label_batch_mask = new_mst_edge_label_batch.to(torch.bool).to(h.device)
                    new_mst_not_edge_label_batch_mask = new_mst_not_edge_label_batch.to(torch.bool).to(h.device)

                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -epsilon
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -epsilon

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 3 * loss_unconstrained + 3 * loss_constrained

    def loss_edges_final_final_final1_GPU_change_infinity_less_mem(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, epsilon, num_edges=500):
        loss_unconstrained = self.loss_edges_infinity(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        # print('loss_unconstrained device: ', loss_unconstrained.device)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                # tx = t.clone().detach()
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0
            # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
            nllloss_func = nn.NLLLoss(reduction='mean')

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
                # 6个节点
                # cost_adj_batch = torch.zeros_like(full_adj)
                cost_adj_batch = np.zeros((num_nodes, num_nodes))

                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                # inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                inverse_full_adj = full_adj == 0
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular)

                # print('pos edge device: ', pos_edge.device)
                # print('neg edge device: ', neg_edges.device)
                # pos_edge = pos_edge.cpu()

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :]

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # take_neg = neg_edges.shape[0]
                # Take the desired number of negative edges, reducing the total edges
                # take_neg = min(neg_edges.shape[0], pos_edge.shape[0] * 2)  # Negative edges are twice the positive edges
                # total_edge = pos_edge.shape[0] + take_neg

                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0).to(h.device)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # Initialize edge label batches efficiently using NumPy
                new_mst_edge_label_batch_np = np.zeros(total_edge, dtype=float)
                new_mst_not_edge_label_batch_np = np.zeros(total_edge, dtype=float)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind].cpu()
                    new_neg_edges = all_edges_[~mask_ind].cpu()
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0].cpu().numpy()  # 第二维度1是精度 所以第一维度0是loss
                    all_edges_ = all_edges_.cpu()
                    all_edges_np = all_edges_.numpy()
                    x, y = all_edges_np.T  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch = np.triu(cost_adj_batch, k=1)

                    ###########################################################################
                    # numpy版本
                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch)
                    # mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch[mst_adj_batch != 0] = 1
                    # Extract the non-zero edges (MST edges) from the csr_matrix
                    mst_rows, mst_cols, _ = find(mst_adj_batch)

                    # Convert MST edges to a set for fast lookup and ensure symmetry
                    mst_edges_set = set(zip(mst_rows, mst_cols))
                    mst_edges_set.update(set(zip(mst_cols, mst_rows)))
                    ###########################################################################
                    # numpy 版本
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    new_pos_edges_set = set(map(tuple, new_pos_edges.tolist()))
                    mask_pos_np = np.array([tuple(edge) in new_pos_edges_set for edge in all_edges_np], dtype=bool)
                    # 计算所有边是否在mst_adj_batch中
                    # mask_mst_np = mst_adj_batch[all_edges_np[:, 0], all_edges_np[:, 1]] == 1
                    mask_mst_np = np.array([tuple(edge) in mst_edges_set for edge in all_edges_np], dtype=bool)
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    new_neg_edges_set = set(map(tuple, new_neg_edges.tolist()))
                    mask_neg_np = np.array([tuple(edge) in new_neg_edges_set for edge in all_edges_np], dtype=bool)

                    ###########################################################################
                    # numpy 版本
                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # Update edge label batches using NumPy for faster bitwise operations
                    new_mst_edge_label_batch_np[mask_pos_np & ~mask_mst_np] = 1
                    # # 更新new_mst_not_edge_label_batch
                    # # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch_np[mask_neg_np & mask_mst_np] = 1
                    new_mst_edge_label_batch_mask = torch.from_numpy(new_mst_edge_label_batch_np).to(torch.bool).to(h.device)
                    new_mst_not_edge_label_batch_mask = torch.from_numpy(new_mst_not_edge_label_batch_np).to(torch.bool).to(h.device)
                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -epsilon
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -epsilon

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 3 * loss_unconstrained + 3 * loss_constrained

    def loss_edges_final_final_final1_GPU_change_infinity_less_mem_same_shuffle(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, epsilon, num_edges=500):
        loss_unconstrained, shuffle_pos_lst, shuffle_neg_lst = self.loss_edges_infinity_same_shuffle(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        # print('loss_unconstrained device: ', loss_unconstrained.device)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                # tx = t.clone().detach()
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0
            # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
            nllloss_func = nn.NLLLoss(reduction='mean')

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                num_nodes = n.shape[0]
                full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
                # 6个节点
                # cost_adj_batch = torch.zeros_like(full_adj)
                cost_adj_batch = np.zeros((num_nodes, num_nodes))

                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                # inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                inverse_full_adj = full_adj == 0
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular)

                # print('pos edge device: ', pos_edge.device)
                # print('neg edge device: ', neg_edges.device)
                # pos_edge = pos_edge.cpu()

                # shuffle edges for undirected edge
                # shuffle = np.random.randn((pos_edge.shape[0])) > 0
                shuffle = shuffle_pos_lst[batch_id]
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :]

                # shuffle edges for undirected edge
                # shuffle = np.random.randn((neg_edges.shape[0])) > 0
                shuffle = shuffle_neg_lst[batch_id]
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                # take_neg = neg_edges.shape[0]
                # Take the desired number of negative edges, reducing the total edges
                # take_neg = min(neg_edges.shape[0], pos_edge.shape[0] * 2)  # Negative edges are twice the positive edges
                # total_edge = pos_edge.shape[0] + take_neg

                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0).to(h.device)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # Initialize edge label batches efficiently using NumPy
                new_mst_edge_label_batch_np = np.zeros(total_edge, dtype=float)
                new_mst_not_edge_label_batch_np = np.zeros(total_edge, dtype=float)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].view(1, -1).repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind].cpu()
                    new_neg_edges = all_edges_[~mask_ind].cpu()
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0].cpu().numpy()  # 第二维度1是精度 所以第一维度0是loss
                    all_edges_ = all_edges_.cpu()
                    all_edges_np = all_edges_.numpy()
                    x, y = all_edges_np.T  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch = np.triu(cost_adj_batch, k=1)

                    ###########################################################################
                    # numpy版本
                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch)
                    # mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch[mst_adj_batch != 0] = 1
                    # Extract the non-zero edges (MST edges) from the csr_matrix
                    mst_rows, mst_cols, _ = find(mst_adj_batch)

                    # Convert MST edges to a set for fast lookup and ensure symmetry
                    mst_edges_set = set(zip(mst_rows, mst_cols))
                    mst_edges_set.update(set(zip(mst_cols, mst_rows)))
                    ###########################################################################
                    # numpy 版本
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    new_pos_edges_set = set(map(tuple, new_pos_edges.tolist()))
                    mask_pos_np = np.array([tuple(edge) in new_pos_edges_set for edge in all_edges_np], dtype=bool)
                    # 计算所有边是否在mst_adj_batch中
                    # mask_mst_np = mst_adj_batch[all_edges_np[:, 0], all_edges_np[:, 1]] == 1
                    mask_mst_np = np.array([tuple(edge) in mst_edges_set for edge in all_edges_np], dtype=bool)
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    new_neg_edges_set = set(map(tuple, new_neg_edges.tolist()))
                    mask_neg_np = np.array([tuple(edge) in new_neg_edges_set for edge in all_edges_np], dtype=bool)

                    ###########################################################################
                    # numpy 版本
                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # Update edge label batches using NumPy for faster bitwise operations
                    new_mst_edge_label_batch_np[mask_pos_np & ~mask_mst_np] = 1
                    # # 更新new_mst_not_edge_label_batch
                    # # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch_np[mask_neg_np & mask_mst_np] = 1
                    new_mst_edge_label_batch_mask = torch.from_numpy(new_mst_edge_label_batch_np).to(torch.bool).to(h.device)
                    new_mst_not_edge_label_batch_mask = torch.from_numpy(new_mst_not_edge_label_batch_np).to(torch.bool).to(h.device)
                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -epsilon
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -epsilon

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 3 * loss_unconstrained + 3 * loss_constrained

    def loss_edges_final_final_final1_GPU_change_infinity_1(self, h, target_nodes, target_edges, indices, epoch,
                                   max_epoch, last_epoch, num_edges=500):
        loss_unconstrained = self.loss_edges_infinity(h=h, target_nodes=target_nodes, target_edges=target_edges, indices=indices,
                                             num_edges=num_edges)

        ######################### constrained ##################################
        """Compute the losses related to the masks: the focal loss and the dice loss.
                   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            #         [0, 5],
            #         [2, 3]], device='cuda:0')]  判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                # tensor([2, 3, 5, 4, 0, 1])
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    # idx=0 k=2
                    t[tx == k] = idx
                new_target_edges.append(t)

            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)
                # cost_adj_batch = torch.zeros_like(full_adj)
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # 创建一个上三角矩阵，但是对角线上的元素设置为 1，而不是默认的 0
                mask_upper_triangular = torch.triu(torch.ones(full_adj.shape[0], full_adj.shape[0]), diagonal=1)
                # 把 full_adj 中值为 0 的元素设置为 1，值为 1 的元素设置为 0
                inverse_full_adj = torch.where(full_adj == 0, torch.tensor(1.), torch.tensor(0.))
                # 在 mask_upper_triangular 和 inverse_full_adj 上做逻辑 AND 操作，只保留上三角矩阵中值为 0 的元素
                zero_upper_triangular = mask_upper_triangular * inverse_full_adj
                # 找到值为 0 的元素的坐标
                pos_edge = torch.nonzero(zero_upper_triangular).to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0])) > 0
                # [False False False False]  在其中选择大于零的位置
                to_shuffle = pos_edge[shuffle, :]
                # tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64) 没有选出来
                pos_edge[shuffle, :] = to_shuffle[:, [1, 0]]
                # 将pos_edge shuffle

                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                # tensor([ 0,  5,  7,  8, 10,  4,  6,  3,  2,  9,  1])

                neg_edges = neg_edges[idx_, :].to(h.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0])) > 0
                # [ True  True False False  True  True  True False False  True  True] 11/7

                to_shuffle = neg_edges[shuffle, :]
                neg_edges[shuffle, :] = to_shuffle[:, [1, 0]]
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                     torch.zeros(take_neg, dtype=torch.long, device=h.device)), 0)

                # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
                nllloss_func = nn.NLLLoss(reduction='mean')
                mst_edge_label_batch = torch.cat(
                    (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                # 这里面用的全是1 因为之后要乘以softmax的第二列
                # edge_labels.append(edge_label_batch)
                # [tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
                # print(epoch, last_epoch, max_epoch)

                # new_mst_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)
                #
                # new_mst_not_edge_label_batch = torch.cat(
                #     (torch.ones(pos_edge.shape[0], dtype=torch.float, device=h.device),
                #      torch.ones(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                new_mst_not_edge_label_batch = torch.cat(
                    (torch.zeros(pos_edge.shape[0], dtype=torch.float, device=h.device),
                     torch.zeros(take_neg, dtype=torch.float, device=h.device)), 0)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature_batch = torch.cat((rearranged_object_token[all_edges_[:, 0], :],
                                                        rearranged_object_token[all_edges_[:, 1], :],
                                                        relation_token[batch_id, ...].repeat(total_edge, 1)), 1).clone()
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)

                    ####################################################################################################
                    # 在假的中取得成果
                    relation_pred_batch_fake = relation_pred_batch.detach()
                    # 得到 e hat
                    ###########################################################################
                    # 首先计算y_hat ij
                    # 所需要的是E hat  和 E（MST）
                    # 其中 E 是由new pos   new neg组成
                    # 当MST=1  但是 E为0也就是new neg的时候  定义为 E+  在 0 * epsilon
                    # 当MST=0  但是 E为1也就是new pos的时候  定义为 E-  在 1 * epsilon
                    # 找出每一行的最大值和对应的索引
                    values, ind = torch.max(relation_pred_batch_fake, dim=1)
                    # 创建一个布尔值的mask，其值表示对应的边是否应该被添加到new_pos_edge_list
                    mask_ind = ind == 1
                    # 使用mask_ind选择对应的边，并创建new_pos_edges和new_neg_edges
                    new_pos_edges = all_edges_[mask_ind]
                    new_neg_edges = all_edges_[~mask_ind]
                    ###########################################################################
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch_fake, dim=-1)
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    x, y = all_edges_.t()  # Transpose to get separate x and y arrays
                    cost_adj_batch[x, y] = cost_pred_batch
                    cost_adj_batch[y, x] = cost_pred_batch
                    cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # for num_pairs in range(all_edges_.shape[0]):
                    #     x, y = all_edges_[num_pairs]
                    #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                    #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
                    # mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()  # 这个情况下是全对称
                    # mst_adj_batch = torch.tensor(mst_adj_batch).to(h.device)

                    # 将numpy数组转换为torch tensor，并移动到GPU上
                    mst_adj_batch = compute_mst_prim(all_edges_, cost_pred_batch).to(h.device)

                    mst_adj_batch[mst_adj_batch != 0] = 1

                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    # 计算所有边是否在new_pos_edges中
                    mask_pos = (all_edges_.unsqueeze(1) == new_pos_edges).all(dim=-1).any(dim=-1)
                    # 计算所有边是否在mst_adj_batch中
                    mask_mst = mst_adj_batch[all_edges_[:, 0], all_edges_[:, 1]] == 1
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    # 计算所有边是否在new_neg_edges中
                    mask_neg = (all_edges_.unsqueeze(1) == new_neg_edges).all(dim=-1).any(dim=-1)
                    ###########################################################################
                    # 现在计算E*+ E*-
                    # 所需要的是E* 真值  和 E（MST）
                    # 其中 E* 是由pos   neg组成
                    # 当E*是pos 但是MST是0  定义为 E*+ 在 0 * epsilon
                    # 当E*是neg 但是MST是1  定义为 E*- 在 1 * epsilon
                    # 在新版中，只对应E*+ E*-对new_mst_edge_label_batch和new_mst_not_edge_label_batch
                    # 进行制约  之外的不进行
                    # 创建一个全为 False 的临时 mask，其长度与 new_mst_not_edge_label_batch 相同
                    # temp_mask = torch.zeros(pos_edge.shape[0] + take_neg, dtype=torch.bool, device=h.device)
                    # # 用 mask_not_in_mst 更新这个临时 mask 的前面部分（对应正边）
                    # # 当E*是pos 但是MST是0  定义为 E*+
                    # temp_mask[:pos_edge.shape[0]] = (mst_adj_batch[pos_edge[:, 0], pos_edge[:, 1]] == 0)
                    # # 当E*是neg 但是MST是1  定义为 E*-
                    # temp_mask[pos_edge.shape[0]:] = (
                    #         mst_adj_batch[neg_edges[:take_neg, 0], neg_edges[:take_neg, 1]] == 1)

                    # 更新new_mst_edge_label_batch
                    # 创建布尔类型条件
                    # 计算 E=1 >>new pos MST = 0  这个就是E-
                    new_mst_edge_label_batch[(mask_pos & ~mask_mst)] = 1
                    # 更新new_mst_not_edge_label_batch
                    # 计算 E=0 >>new neg MST = 1  这个就是E+
                    new_mst_not_edge_label_batch[(mask_neg & mask_mst)] = 1

                    new_mst_edge_label_batch_mask = new_mst_edge_label_batch.to(torch.bool)
                    new_mst_not_edge_label_batch_mask = new_mst_not_edge_label_batch.to(torch.bool)

                    ###########################################################################
                    relation_pred_batch_true = relation_pred_batch.clone()
                    # 使用布尔掩码来选择需要修改的行
                    relation_pred_batch_true[new_mst_edge_label_batch_mask, 1] = -10
                    relation_pred_batch_true[new_mst_not_edge_label_batch_mask, 0] = -10

                    # relation_pred_batch_true[:, 1] = relation_pred_batch_true[:, 1] * new_mst_edge_label_batch
                    # relation_pred_batch_true[:, 0] = relation_pred_batch_true[:, 0] * new_mst_not_edge_label_batch

                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch_true, dim=-1)

                    ##### 控制界限
                    if relation_pred_softmax_batch_true.lt(MIN_TEST).any():
                        mask = relation_pred_softmax_batch_true < MIN_TEST
                        relation_pred_softmax_batch_true = relation_pred_softmax_batch_true.clone()  # 创建一个新的变量
                        relation_pred_softmax_batch_true[mask] = MIN_TEST  # 在新的变量上执行操作

                    #####
                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    # 只包括 E*+ 和 E*- 的损失
                    # print(111111111111111)
                    # print("nlloss_batch: ", nlloss_batch.shape)
                    # print("loss_mask: ", loss_mask.shape)
                    # print("loss_mask sum: ", loss_mask.sum())
                    # nlloss = (nlloss_batch).sum() / max(total_edge, 1)
                    loss = loss + nlloss_batch

                else:
                    relation_feature_batch = torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]), 1)
                    relation_pred_batch = self.net.module.relation_embed(relation_feature_batch)
                    # relation_pred_softmax_batch = torch.softmax(relation_pred_batch, dim=-1)
                    # relation_pred_softmax_batch = relation_pred_batch.clone().detach().softmax(-1)
                    relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1).detach().cpu()
                    ###########################################################################
                    cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
                    for num_pairs in range(all_edges_.shape[0]):
                        x, y = all_edges_[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)

                    for pos_pairs in range(pos_edge.shape[0]):
                        x, y = pos_edge[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            # mst_edge_label_batch[pos_pairs] = \
                            #     1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch-last_epoch)

                            mst_edge_label_batch[pos_pairs] = 0.01
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1

                    ###########################################################################
                    relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1)
                    relation_pred_softmax_batch_true[:, 1] = relation_pred_softmax_batch_true[:,
                                                             1] * mst_edge_label_batch

                    relation_pred_softmax_batch_true[:, 0] = relation_pred_softmax_batch_true[:, 0] + \
                                                             relation_pred_softmax_batch_true[:, 1] * (
                                                                     1 - mst_edge_label_batch)

                    relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
                    nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
                    loss = loss + nlloss_batch


        except Exception as e:
            print(e)
            pdb.set_trace()
        loss_constrained = loss / h.shape[0]
        # print("un   cons: ", loss_unconstrained, loss_constrained)
        return 1 * loss_unconstrained + 1 * loss_constrained

    def loss_gnnes(self, h, target_nodes, target_edges, indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # 判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    t[tx == k] = idx
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                # neg_edges = torch.nonzero(torch.triu(full_adj))
                neg_edges = torch.nonzero(full_adj).to(torch.long).t().contiguous()
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])
                pos_edge_1 = []
                all_full_adj = []
                for row in range(full_adj.shape[0]):
                    for col in range(full_adj.shape[1]):
                        all_full_adj.append([row, col])
                        if full_adj[row, col] == 0:
                            pos_edge_1.append([row, col])
                all_full_adj = torch.tensor(all_full_adj, device=h.device, dtype=torch.long).t().contiguous()
                pos_edge_1 = torch.tensor(pos_edge_1, device=h.device, dtype=torch.long).t().contiguous()

                # for col in range(pos_edge_1.shape[1]):
                #     pos = pos_edge_1[:, col]
                #     for neg_col in range(neg_edges.shape[1]):
                #         if pos == neg_edges[:, neg_col]:
                #             print("@" * 20)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature = torch.cat([rearranged_object_token,  # 6 256 + 1 256 *6
                                                  relation_token[batch_id, ...].repeat(rearranged_object_token.shape[0],
                                                                                       1)], 1)
                    z_train = self.net.module.GAE_model.encode((relation_feature), all_full_adj)  # 6 * 512 >> 6*256
                    gala_out = self.net.module.gala_model(z_train, all_full_adj)  # 6 256 >> 6 2(x, y)
                    gala_loss_type = MSELoss(reduction='mean')
                    gala_loss = gala_loss_type(n, gala_out)
                    train_loss = self.net.module.GAE_model.recon_loss(z_train, pos_edge_1, neg_edges)  # 2 4488

                    loss = loss + gala_loss + train_loss



                else:
                    relation_feature = rearranged_object_token
                    z_train = self.net.module.GAE_model.encode((relation_feature), all_full_adj)  # 6 * 512 >> 6*256
                    gala_out = self.net.module.gala_model(z_train, all_full_adj)  # 6 256 >> 6 2(x, y)
                    gala_loss_type = MSELoss(reduction='mean')
                    gala_loss = gala_loss_type(n, gala_out)
                    train_loss = self.net.module.GAE_model.recon_loss(z_train, pos_edge_1, neg_edges)  # 2 4488
                    loss = loss + gala_loss + train_loss

        except Exception as e:
            print(e)
            pdb.set_trace()

        # print(loss.device)
        # print(torch.isnan(loss).sum())
        # if torch.isnan(loss).sum() != 0:
        #     print("$"*20)

        return loss / h.shape[0]

    def loss_gnnes_mst(self, h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # h 2 21 256
        # target_nodes
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
        # target_edges
        # [tensor([[0, 1],
        #         [1, 3],
        #         [2, 4],
        #         [4, 5]], device='cuda:0'), tensor([[0, 1],
        #         [0, 2],
        #         [0, 4],
        #         [0, 5],
        #         [2, 3]], device='cuda:0')]
        # indices
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))]
        # num_edges
        # 40
        try:
            # all token except the last one is object token
            object_token = h[..., :self.obj_token, :]
            # 2 20 256

            # last token is relation token
            if self.rln_token > 0:  # 1
                relation_token = h[..., self.obj_token:self.rln_token + self.obj_token, :]
                # 2 1 256

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                            zip(target_edges, indices)]
            # target_edges按照indices（_, 1）进行确认是否都在indices的范围之内
            target_edges = [torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(h.device) for
                            t in
                            target_edges]
            # 判断输入的边的个数要大于零
            new_target_edges = []  # 按照indices排列tgt_edges
            for t, (_, i) in zip(target_edges, indices):
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    t[tx == k] = idx
                new_target_edges.append(t)

            # [tensor([[4, 5],
            #         [5, 1],
            #         [0, 3],
            #         [3, 2]], device='cuda:0'), tensor([[3, 4],
            #         [3, 0],
            #         [3, 1],
            #         [3, 5],
            #         [0, 2]], device='cuda:0')]

            # all_edges = []
            loss = 0.0

            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                # batch_id  0
                # pos_edge
                # tensor([[4, 5],
                #         [5, 1],
                #         [0, 3],
                #         [3, 2]], device='cuda:0')
                # n
                # tensor([[0.7763, 0.0778],
                #         [0.5247, 0.0943],
                #         [1.0000, 0.5104],
                #         [0.0727, 0.0000],
                #         [0.7931, 0.4840],
                #         [0.2609, 0.6906]], device='cuda:0')
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]
                # print(indices[batch_id][0])
                # tensor([ 7,  8, 12, 13, 18, 19])
                # 按照这个顺序从20个选出需要的6个  >>> 6 256

                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(torch.ones(n.shape[0]))
                cost_adj_batch = torch.zeros_like(full_adj, device=h.device)

                # 6个节点
                # tensor([[0., 1., 1., 1., 1., 1.],
                #         [1., 0., 1., 1., 1., 1.],
                #         [1., 1., 0., 1., 1., 1.],
                #         [1., 1., 1., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 1.],
                #         [1., 1., 1., 1., 1., 0.]])
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [1., 0., 1., 1., 1., 0.],
                #         [1., 1., 0., 0., 1., 1.],
                #         [0., 1., 0., 0., 1., 1.],
                #         [1., 1., 1., 1., 0., 0.],  (4 5)
                #         [1., 0., 1., 1., 0., 0.]]) (5 4) 对称设置为0
                # neg_edges = torch.nonzero(torch.triu(full_adj))
                neg_edges = torch.nonzero(full_adj).to(torch.long).t().contiguous()
                # 不是0的地方就是neg
                # print(torch.triu(full_adj))
                # tensor([[0., 1., 1., 0., 1., 1.],
                #         [0., 0., 1., 1., 1., 0.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 1., 1.],
                #         [0., 0., 0., 0., 0., 0.],
                #         [0., 0., 0., 0., 0., 0.]])
                # tensor([[0, 1],
                #         [0, 2],
                #         [0, 4],
                #         [0, 5],
                #         [1, 2],
                #         [1, 3],
                #         [1, 4],
                #         [2, 4],
                #         [2, 5],
                #         [3, 4],
                #         [3, 5]])
                pos_edge_1 = []
                all_full_adj = []
                for row in range(full_adj.shape[0]):
                    for col in range(full_adj.shape[1]):
                        all_full_adj.append([row, col])
                        if full_adj[row, col] == 0:
                            pos_edge_1.append([row, col])
                all_full_adj = torch.tensor(all_full_adj, device=h.device, dtype=torch.long).t().contiguous()
                pos_edge_1 = torch.tensor(pos_edge_1, device=h.device, dtype=torch.long).t().contiguous()
                mst_edge_label_batch = torch.ones(all_full_adj.shape[1], dtype=torch.float, device=h.device)

                # for col in range(pos_edge_1.shape[1]):
                #     pos = pos_edge_1[:, col]
                #     for neg_col in range(neg_edges.shape[1]):
                #         if pos == neg_edges[:, neg_col]:
                #             print("@" * 20)

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature = torch.cat([rearranged_object_token,  # 6 256 + 1 256 *6
                                                  relation_token[batch_id, ...].repeat(rearranged_object_token.shape[0],
                                                                                       1)], 1)
                    z_train = self.net.module.GAE_model.encode((relation_feature), all_full_adj)  # 6 * 512 >> 6*256
                    gala_out = self.net.module.gala_model(z_train, all_full_adj)  # 6 256 >> 6 2(x, y)
                    gala_loss_type = MSELoss(reduction='mean')
                    gala_loss = gala_loss_type(n, gala_out)
                    # print(1111111111111111111111)
                    # print(mst_edge_label_batch)
                    # print(all_full_adj)

                    # print(z_train)
                    # print(pos_edge_1)
                    # print(neg_edges)
                    link_logits, link_labels = self.net.module.GAE_model.recon_label_logits(z_train, pos_edge_1,
                                                                                            neg_edges.to(
                                                                                                h.device))  # 2 4488
                    ###########################################################################
                    cost_pred_batch = 1 - link_logits.detach().cpu()
                    # cost_pred_batch = 1 - link_logits.detach()
                    for num_pairs in range(all_full_adj.shape[1]):
                        x, y = all_full_adj[:, num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    # print(cost_pred_batch)
                    # print(link_logits)
                    # print(cost_adj_batch)
                    # print(all_full_adj)
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()
                    # print(mst_adj_batch)

                    for pos_pairs in range(pos_edge_1.shape[1]):
                        x, y = pos_edge_1[:, pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            mst_edge_label_batch[pos_pairs] = \
                                1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch - last_epoch)
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1
                    ###########################################################################
                    relation_pred_softmax_batch_true = link_logits.clone()
                    relation_pred_softmax_batch_true = relation_pred_softmax_batch_true * mst_edge_label_batch
                    # print(11111111111111111111111111111111111111111)
                    train_loss = F.binary_cross_entropy_with_logits(relation_pred_softmax_batch_true, link_labels,
                                                                    reduction='mean')
                    # print(train_loss)
                    # print(gala_loss)
                    loss = loss + gala_loss + train_loss



                else:
                    relation_feature = rearranged_object_token
                    z_train = self.net.module.GAE_model.encode((relation_feature), all_full_adj)  # 6 * 512 >> 6*256
                    gala_out = self.net.module.gala_model(z_train, all_full_adj)  # 6 256 >> 6 2(x, y)
                    gala_loss_type = MSELoss(reduction='mean')
                    gala_loss = gala_loss_type(n, gala_out)
                    link_logits, link_labels = self.net.module.GAE_model.recon_label_logits(z_train, pos_edge_1,
                                                                                            neg_edges)  # 2 4488
                    ###########################################################################
                    cost_pred_batch = 1 - link_logits.detach().cpu()
                    for num_pairs in range(all_full_adj.shape[0]):
                        x, y = all_full_adj[num_pairs]
                        cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
                        cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
                    cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

                    # cost_adj_batch = assignment_with_nan(cost_adj_batch.cpu().numpy().copy())
                    # mst_adj_batch = minimum_spanning_tree(cost_adj_batch)

                    mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
                    mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

                    for pos_pairs in range(pos_edge_1.shape[0]):
                        x, y = pos_edge_1[pos_pairs]
                        if mst_adj_batch[x, y] == 0:  # 不是树
                            mst_edge_label_batch[pos_pairs] = \
                                1 - ((1 - 0.01) / (max_epoch - last_epoch)) * (epoch - last_epoch)
                            # 对于不是树 但是被选中的要变成0.01 是树不变 1
                    ###########################################################################
                    relation_pred_softmax_batch_true = link_logits.clone()
                    relation_pred_softmax_batch_true = relation_pred_softmax_batch_true * mst_edge_label_batch
                    train_loss = F.binary_cross_entropy_with_logits(relation_pred_softmax_batch_true, link_labels,
                                                                    reduction='mean')
                    loss = loss + gala_loss + train_loss

        except Exception as e:
            print(e)
            pdb.set_trace()

        # print(loss.device)
        # print(torch.isnan(loss).sum())
        # if torch.isnan(loss).sum() != 0:
        #     print("$"*20)
        return loss / h.shape[0]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, h, out, target, epoch, max_epoch, last_epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(out, target)
        # [(tensor([ 7,  8, 12, 13, 18, 19]), tensor([2, 3, 5, 4, 0, 1])),
        # (tensor([ 9, 12, 13, 14, 18, 19]), tensor([2, 4, 3, 0, 1, 5]))] 预测20个中的第九个对应答案6个的第二个
        # print(indices)

        losses = {}
        # print(out['pred_logits'][0, :5, :])
        # print(out['pred_nodes'][0, :5, :])
        losses['class'] = self.loss_class(out['pred_logits'], indices)
        losses['nodes'] = self.loss_nodes(out['pred_nodes'][..., :2], target['nodes'], indices)
        losses['boxes'] = self.loss_boxes(out['pred_nodes'], target['nodes'], indices)

        if self.use_mst_train and not self.use_gnn:
            # losses['edges'] = self.loss_edges_final_final_final1_GPU(h, target['nodes'], target['edges'], indices, epoch=epoch,
            #                                           max_epoch=max_epoch, last_epoch=last_epoch)
            losses['edges'] = self.loss_edges_final_final_final1_GPU_change_infinity_less_mem(h, target['nodes'],
                                                                                     target['edges'], indices,
                                                                                     epoch=epoch,
                                                                                     max_epoch=max_epoch,
                                                                                     last_epoch=last_epoch,
                                                                                     epsilon=10)
        elif self.use_gnn and not self.use_mst_train:
            losses['edges'] = self.loss_gnnes(h, target['nodes'], target['edges'], indices)
        elif self.use_gnn and self.use_mst_train:
            losses['edges'] = self.loss_gnnes_mst(h, target['nodes'], target['edges'], indices, epoch=epoch,
                                                  max_epoch=max_epoch, last_epoch=last_epoch)
        else:
            # losses['edges'] = self.loss_edges(h, target['nodes'], target['edges'], indices)
            losses['edges'] = self.loss_edges_infinity_6(h, target['nodes'], target['edges'], indices)


        losses['cards'] = self.loss_cardinality(out['pred_logits'], indices)
        losses['total'] = sum([losses[key] * self.weight_dict[key] for key in self.losses])

        return losses

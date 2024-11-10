import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import itertools
import torch.distributed as dist
import torch.autograd as autograd

import networkx as nx
import numpy as np

def compute_mst_prim(node_pairs_valid, cost_pred_batch):
    # 创建 NetworkX 图
    G = nx.Graph()

    # 将节点对和对应的成本添加到图中
    # 使用 tensor 的 numpy 表示来提高性能
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    # 使用 NumPy 数组的批处理能力添加边
    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    # 使用 Prim 算法计算最小生成树
    # mst_edges = nx.minimum_spanning_edges(G, algorithm="prim", data=False)
    mst_edges = nx.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    mst_edges = list(mst_edges)

    # 创建 MST 的邻接矩阵
    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    # 将 numpy 数组转换回 torch Tensor
    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch

def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False):
    # all token except the last one is object token
    # 2 21 256    dict  model   20 1   F  True
    object_token = h[..., :obj_token, :]
    # 2 20 256
    # last token is relation token
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]
        # 2 1 256

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()
    # 返回指定维度最大值的序号
    # 第一个>第二个 0
    # 第二个>第一个 1
    # 'pred_logits':
    #  tensor([[[4.2759, -4.9492],  0
    #          [ 4.7401, -5.0336],  0
    #          [-4.2770,  4.9252],  1
    #          [ 4.6597, -4.3927],  0
    #          [ 3.0228, -2.4876],  1
    #          [ 2.3914, -2.8800],  1
    #          [-4.4479,  4.6817],  0
    #          ...
    #         [[ 3.5328, -3.9187],  0
    #          [ 2.1835, -1.8711],  0
    #          [-3.5960,  4.7048],  1
    #          [ 2.2582, -2.0884],  0
    #          [-3.5567,  4.6325],  1
    #          [ 3.8503, -4.4690],  0
    #          [-5.4784,  6.0805],  1
    #          ....
    # tensor([[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]],
    #        device='cuda:0')

    # apply nms on valid tokens
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            valid_token_id = torch.nonzero(token).squeeze(1)

            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # 0 <= x1 < x2 and 0 <= y1 < y2 has to be fulfilled
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            ids2keep = batched_nms(
                boxes=valid_nodes * 1000, scores=valid_scores, idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            # print(valid_nodes.shape[0] - ids2keep.shape[0])

            valid_token_nms[idx][valid_token_id_nms] = 1
        valid_token = valid_token_nms

    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []

        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        # batch_id 0
        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        # tensor([ 2,  6,  8, 12, 13, 16, 18], device='cuda:0')

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())
        # 前2个为坐标  后面的为0.2 0.2 所以不需要
        # [tensor([[0.1841, 0.0691],
        #         [0.4373, 0.5390],
        #         [0.3560, 0.3794],
        #         [0.5107, 0.1138],
        #         [0.4928, 0.8826],
        #         [0.2771, 0.3296],
        #         [0.5259, 0.2376]], device='cuda:0')]

        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            # [array([[0.18412243, 0.06907109, 0.1967704 , 0.19651298],
            #        [0.43725446, 0.5389896 , 0.19494587, 0.19415428],
            #        [0.3559776 , 0.3794247 , 0.19489756, 0.19481266],
            #        [0.51065665, 0.11383494, 0.19533527, 0.1966407 ],
            #        [0.49280432, 0.88256013, 0.19605483, 0.19701621],
            #        [0.2771306 , 0.32960707, 0.19484201, 0.1947407 ],
            #        [0.52590996, 0.23761915, 0.19810419, 0.19754113]], dtype=float32)]  加入了0.2 0.2
            pred_nodes_boxes_score.append(out['pred_logits'].softmax(-1)[
                                              batch_id, node_id, 1].detach().cpu().numpy())  # TODO: generalize over multi-class
            # [array([0.99989915, 0.99989164, 0.99963665, 0.99967265, 0.99997485,
            #        0.9995617 , 0.99777764], dtype=float32)]
            #        对最后一个维度取softmax 得到每一种概率 然后只取 指定的batch 指定的node_id的第二维度的值  class：选择 【0，1】不选择【1，0】
            # 所以里面的值都很接近一

            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())
            # [array([1, 1, 1, 1, 1, 1, 1], dtype=int64)]

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:

            # all possible node pairs in all token ordering
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            # [[tensor(2, device='cuda:0'), tensor(6, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(8, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(12, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(13, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(16, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(18, device='cuda:0')],
            # [tensor(6, device='cuda:0'), tensor(8, device='cuda:0')],
            # [tensor(6, device='cuda:0'), tensor(12, device='cuda:0')], ...
            node_pairs = list(map(list, zip(*node_pairs)))
            # [[tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(12, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0')], [tensor(6, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(18, device='cuda:0')]]
            # [2 2 2 2 2 6 6...] 21
            # [6 8 12 13 16 18 ...] 21
            # 21个是怎么得到的： [ 2,  6,  8, 12, 13, 16, 18]  2*6 2*8 2*12 ...6个 对于6来说 6*8 6*12...
            # 21 = 6+5+4+2+3+2+1 = 21

            # node pairs in valid token order
            node_pairs_valid = torch.tensor([list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))])
            # tensor([[0, 1],
            #         [0, 2],
            #         [0, 3],
            #         [0, 4],
            #         [0, 5],
            #         [0, 6],  6
            #         [1, 2],
            #         [1, 3],
            #         [1, 4],
            #         [1, 5],
            #         [1, 6], 5
            #         [2, 3],
            #         [2, 4],
            #         [2, 5],
            #         [2, 6], 4
            #         [3, 4],
            #         [3, 5],
            #         [3, 6], 3
            #         [4, 5],
            #         [4, 6], 2
            #         [5, 6]])1
            # concatenate valid object pairs relation feature
            if rln_token > 0:
                relation_feature1 = torch.cat((object_token[batch_id, node_pairs[0], :],
                                               object_token[batch_id, node_pairs[1], :],
                                               relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)), 1)
                # 第一个是将node1 node2这种顺序加入 然后将rln_token复制21倍 再cat 》21 768
                relation_feature2 = torch.cat((object_token[batch_id, node_pairs[1], :],
                                               object_token[batch_id, node_pairs[0], :],
                                               relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)), 1)
                # 第二个是将node2 node1这种顺序加入 然后将rln_token复制21倍 再cat 》21 768
            else:
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :], object_token[batch_id, node_pairs[1], :]), 1)
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :], object_token[batch_id, node_pairs[0], :]), 1)

            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            # relation_feature1_idx_ = torch.randperm(relation_feature1.shape[0])
            # relation_feature1_shuffle = relation_feature1[relation_feature1_idx_, :].to(relation_feature1.device)
            # relation_feature1_shuffle_pred = model.relation_embed(relation_feature1_shuffle).detach()

            # tensor([[6.8234, -5.8577],
            #         [-5.7044, 5.5189],
            #         [7.3179, -6.6655],
            #         [16.2090, -13.3791],
            #         [6.0530, -4.4774],
            #         [3.8634, -3.5972],
            #         [-6.7858, 6.4497],
            #         [4.6484, -4.2432],
            #         [-8.0337, 7.2780],
            #         [5.8594, -5.9227],
            #         [-5.7051, 5.5116],
            #         [3.7694, -3.3850],
            #         [9.9119, -8.7369],
            #         [-5.1872, 4.6661],
            #         [3.1310, -3.1254],
            #         [16.3112, -12.4753],
            #         [9.2421, -5.3801],
            #         [-4.6888, 4.3664],
            #         [14.5805, -14.7794],
            #         [10.0542, -9.4428],
            #         [4.5389, -4.6362]], device='cuda:0')
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()
            # tensor([[  4.8644,  -5.4507],
            #         [ -5.2172,   5.0186],
            #         [  9.4442,  -6.4797],
            #         [ 14.6816, -15.4302],
            #         [  6.0261,  -5.6813],
            #         [  5.2363,  -3.5658],
            #         [ -6.2179,   5.8572],
            #         [  5.7630,  -4.6627],
            #         [ -8.1400,   7.6413],
            #         [  7.1933,  -6.3217],
            #         [ -7.1334,   7.0079],
            #         [  5.5766,  -4.0371],
            #         [  9.6856,  -8.7855],
            #         [ -4.3368,   3.7852],
            #         [  7.4150,  -5.2945],
            #         [ 14.3405, -14.6813],
            #         [  6.1081,  -5.8770],
            #         [ -3.9738,   3.7016],
            #         [ 17.2839, -14.5458],
            #         [ 11.3758,  -9.1620],
            #         [  7.0686,  -4.6173]], device='cuda:0')
            # relation_pred1 = F.relu(relation_pred1)
            # relation_pred2 = F.relu(relation_pred2)
            relation_pred = (relation_pred1 + relation_pred2) / 2.0
            # tensor([[  5.8439,  -5.6542],
            #         [ -5.4608,   5.2688],
            #         [  8.3810,  -6.5726],
            #         [ 15.4453, -14.4047],
            #         [  6.0396,  -5.0793],
            #         [  4.5499,  -3.5815],
            #         [ -6.5018,   6.1535],
            #         [  5.2057,  -4.4530],
            #         [ -8.0869,   7.4596],
            #         [  6.5264,  -6.1222],
            #         [ -6.4193,   6.2597],
            #         [  4.6730,  -3.7110],
            #         [  9.7988,  -8.7612],
            #         [ -4.7620,   4.2256],
            #         [  5.2730,  -4.2099],
            #         [ 15.3258, -13.5783],
            #         [  7.6751,  -5.6286],
            #         [ -4.3313,   4.0340],
            #         [ 15.9322, -14.6626],
            #         [ 10.7150,  -9.3024],
            #         [  5.8037,  -4.6268]], device='cuda:0')

            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1).cpu().numpy()
            # print(torch.argmax(relation_pred, -1))
            # tensor([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], device='cuda:0')
            # print((node_id))
            # tensor([ 2,  6,  8, 12, 13, 16, 18], device='cuda:0')
            pred_edges.append(node_pairs_valid[pred_rel].cpu().numpy())
            # [array([[0, 2],
            #        [1, 2],
            #        [1, 4],
            #        [1, 6],
            #        [2, 5],
            #        [3, 6]], dtype=int64)]

            if map_:
                pred_edges_boxes_score.append(relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy())
                # [array([0.99997807, 0.9999968 , 0.9999999 , 0.9999969 , 0.99987507, 0.99976724], dtype=float32)]
                pred_edges_boxes_class.append(torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy())
                # [array([1, 1, 1, 1, 1, 1], dtype=int64)]
        else:
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    if map_:
        return pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score, pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class
    else:
        return pred_nodes, pred_edges

def relation_infer_mst(h, out, net, obj_token, rln_token, nms=False, map_=False):
    # all token except the last one is object token
    # 'pred_nodes':
    # tensor([[[0.7238, 0.1392, 0.1952, 0.1962],
    #          [0.4835, 0.0991, 0.1947, 0.1953],
    #          [0.1841, 0.0691, 0.1968, 0.1965],
    #          [0.2917, 0.1657, 0.2023, 0.1995],
    #          [0.1786, 0.2498, 0.1958, 0.1939],
    #          [0.3548, 0.3452, 0.1995, 0.1996],
    #          [0.4373, 0.5390, 0.1949, 0.1942],
    #          [0.7165, 0.0849, 0.1981, 0.1984],
    #          [0.3560, 0.3794, 0.1949, 0.1948],
    #          [0.3610, 0.1147, 0.1888, 0.1893],
    #          [0.2132, 0.1397, 0.1995, 0.1987],
    #          [0.5068, 0.2446, 0.1969, 0.1979],
    #          [0.5107, 0.1138, 0.1953, 0.1966],
    #          [0.4928, 0.8826, 0.1961, 0.1970],
    #          [0.3503, 0.2831, 0.1963, 0.1963],
    #          [0.4440, 0.1837, 0.2069, 0.2058],
    #          [0.2771, 0.3296, 0.1948, 0.1947],
    #          [0.8024, 0.0798, 0.2014, 0.2011],
    #          [0.5259, 0.2376, 0.1981, 0.1975],
    #          [0.2815, 0.3279, 0.2002, 0.2009]],
    #         [[0.6836, 0.2759, 0.1979, 0.1985],
    #          [0.4160, 0.2413, 0.2074, 0.2042],
    #          [0.2138, 0.1994, 0.1946, 0.1947],
    #          [0.3712, 0.1612, 0.2030, 0.2010],
    #          [0.3974, 0.2684, 0.1943, 0.1955],
    #          [0.6089, 0.3865, 0.1902, 0.1924],
    #          [0.5396, 0.6570, 0.1961, 0.1966],
    #          [0.6644, 0.0989, 0.2000, 0.1989],
    #          [0.4631, 0.4353, 0.1952, 0.1945],
    #          [0.3708, 0.1896, 0.1978, 0.1972],
    #          [0.3530, 0.1280, 0.1955, 0.1963],
    #          [0.8104, 0.4228, 0.2088, 0.2126],
    #          [0.7645, 0.3696, 0.2109, 0.2123],
    #          [0.4950, 0.8844, 0.1958, 0.1968],
    #          [0.3818, 0.1117, 0.1964, 0.1975],
    #          [0.8104, 0.0487, 0.1946, 0.1948],
    #          [0.1969, 0.2158, 0.1977, 0.1967],
    #          [0.8167, 0.1104, 0.2093, 0.2096],
    #          [0.7700, 0.3660, 0.1953, 0.1944],
    #          [0.9047, 0.1945, 0.1928, 0.1957]]], device='cuda:0')}
    # 2 21 256    dict  model   20 1   F  True
    object_token = h[..., :obj_token, :]
    # 2 20 256
    # last token is relation token
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]
        # 2 1 256

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()
    # 返回指定维度最大值的序号
    # 第一个>第二个 0
    # 第二个>第一个 1
    # 'pred_logits':
    #  tensor([[[4.2759, -4.9492],  0
    #          [ 4.7401, -5.0336],  0
    #          [-4.2770,  4.9252],  1
    #          [ 4.6597, -4.3927],  0
    #          [ 3.0228, -2.4876],  1
    #          [ 2.3914, -2.8800],  1
    #          [-4.4479,  4.6817],  0
    #          ...
    #         [[ 3.5328, -3.9187],  0
    #          [ 2.1835, -1.8711],  0
    #          [-3.5960,  4.7048],  1
    #          [ 2.2582, -2.0884],  0
    #          [-3.5567,  4.6325],  1
    #          [ 3.8503, -4.4690],  0
    #          [-5.4784,  6.0805],  1
    #          ....
    # tensor([[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]],
    #        device='cuda:0')

    # apply nms on valid tokens
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            valid_token_id = torch.nonzero(token).squeeze(1)

            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # 0 <= x1 < x2 and 0 <= y1 < y2 has to be fulfilled
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            ids2keep = batched_nms(
                boxes=valid_nodes * 1000, scores=valid_scores, idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            # print(valid_nodes.shape[0] - ids2keep.shape[0])

            valid_token_nms[idx][valid_token_id_nms] = 1
        valid_token = valid_token_nms

    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []

        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        # batch_id 0
        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        # tensor([ 2,  6,  8, 12, 13, 16, 18], device='cuda:0')
        cost_adj_batch = torch.ones((node_id.shape[0], node_id.shape[0])).to(h.device) * 9999

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())
        # 前2个为坐标  后面的为0.2 0.2 所以不需要
        # [tensor([[0.1841, 0.0691],
        #         [0.4373, 0.5390],
        #         [0.3560, 0.3794],
        #         [0.5107, 0.1138],
        #         [0.4928, 0.8826],
        #         [0.2771, 0.3296],
        #         [0.5259, 0.2376]], device='cuda:0')]

        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            # [array([[0.18412243, 0.06907109, 0.1967704 , 0.19651298],
            #        [0.43725446, 0.5389896 , 0.19494587, 0.19415428],
            #        [0.3559776 , 0.3794247 , 0.19489756, 0.19481266],
            #        [0.51065665, 0.11383494, 0.19533527, 0.1966407 ],
            #        [0.49280432, 0.88256013, 0.19605483, 0.19701621],
            #        [0.2771306 , 0.32960707, 0.19484201, 0.1947407 ],
            #        [0.52590996, 0.23761915, 0.19810419, 0.19754113]], dtype=float32)]  加入了0.2 0.2
            pred_nodes_boxes_score.append(out['pred_logits'].softmax(-1)[
                                              batch_id, node_id, 1].detach().cpu().numpy())  # TODO: generalize over multi-class
            # [array([0.99989915, 0.99989164, 0.99963665, 0.99967265, 0.99997485,
            #        0.9995617 , 0.99777764], dtype=float32)]
            #        对最后一个维度取softmax 得到每一种概率 然后只取 指定的batch 指定的node_id的第二维度的值  class：选择 【0，1】不选择【1，0】
            # 所以里面的值都很接近一

            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())
            # [array([1, 1, 1, 1, 1, 1, 1], dtype=int64)]

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:

            # all possible node pairs in all token ordering
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            # [[tensor(2, device='cuda:0'), tensor(6, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(8, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(12, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(13, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(16, device='cuda:0')],
            # [tensor(2, device='cuda:0'), tensor(18, device='cuda:0')],
            # [tensor(6, device='cuda:0'), tensor(8, device='cuda:0')],
            # [tensor(6, device='cuda:0'), tensor(12, device='cuda:0')], ...
            node_pairs = list(map(list, zip(*node_pairs)))
            # [[tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(2, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(6, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(12, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0')], [tensor(6, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(8, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(12, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(13, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(16, device='cuda:0'), tensor(18, device='cuda:0'), tensor(18, device='cuda:0')]]
            # [2 2 2 2 2 6 6...] 21
            # [6 8 12 13 16 18 ...] 21
            # 21个是怎么得到的： [ 2,  6,  8, 12, 13, 16, 18]  2*6 2*8 2*12 ...6个 对于6来说 6*8 6*12...
            # 21 = 6+5+4+2+3+2+1 = 21

            # node pairs in valid token order
            node_pairs_valid = torch.tensor([list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))])
            # tensor([[0, 1],
            #         [0, 2],
            #         [0, 3],
            #         [0, 4],
            #         [0, 5],
            #         [0, 6],  6
            #         [1, 2],
            #         [1, 3],
            #         [1, 4],
            #         [1, 5],
            #         [1, 6], 5
            #         [2, 3],
            #         [2, 4],
            #         [2, 5],
            #         [2, 6], 4
            #         [3, 4],
            #         [3, 5],
            #         [3, 6], 3
            #         [4, 5],
            #         [4, 6], 2
            #         [5, 6]])1
            # concatenate valid object pairs relation feature
            node_pairs_valid_dict = {}
            for num in range(node_pairs_valid.shape[0]):
                node_pair = node_pairs_valid[num]
                node_pairs_valid_dict[tuple(node_pair.cpu().numpy().tolist())] = num
            if rln_token > 0:
                relation_feature1 = torch.cat((object_token[batch_id, node_pairs[0], :],
                                               object_token[batch_id, node_pairs[1], :],
                                               relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)), 1)
                # 第一个是将node1 node2这种顺序加入 然后将rln_token复制21倍 再cat 》21 768
                relation_feature2 = torch.cat((object_token[batch_id, node_pairs[1], :],
                                               object_token[batch_id, node_pairs[0], :],
                                               relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)), 1)
                # 第二个是将node2 node1这种顺序加入 然后将rln_token复制21倍 再cat 》21 768
            else:
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :], object_token[batch_id, node_pairs[1], :]), 1)
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :], object_token[batch_id, node_pairs[0], :]), 1)

            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()
            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1).detach()
            cost_pred_batch = relation_pred_softmax_batch[:, 0]  # 第二维度1是精度 所以第一维度0是loss
            # accuracy_pred_batch = relation_pred_softmax_batch[:, 1]  # 第二维度1是精度 所以第一维度0是loss

            # x, y = node_pairs_valid.t()  # Transpose to get separate x and y arrays
            # cost_adj_batch[x, y] = cost_pred_batch
            # cost_adj_batch[y, x] = cost_pred_batch
            # # cost_adj_batch *= local_distance_adj_batch
            # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)
            # mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

            x, y = node_pairs_valid.t()  # Transpose to get separate x and y arrays
            cost_adj_batch[x, y] = cost_pred_batch
            cost_adj_batch[y, x] = cost_pred_batch
            cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)
            # cost_adj_batch *= local_distance_adj_batch

            # mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
            # mst_adj_batch = torch.tensor(mst_adj_batch.toarray())
            # mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

            mst_adj_batch = compute_mst_prim(node_pairs_valid, cost_pred_batch)

            # for num_pairs in range(node_pairs_valid.shape[0]):  # 21 2
            #     x, y = node_pairs_valid[num_pairs]
            #     cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
            #     cost_adj_batch[y, x] = cost_pred_batch[num_pairs]
            #
            # mst_adj_batch = prims_mst(cost_adj=cost_adj_batch)
            # mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

            # 直接获取非零元素的坐标
            mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)

            # 转换为 numpy 数组并添加到 pred_edges 列表中
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            # # 转换为 numpy 数组（如果需要）
            # pred_edges = mst_tree_selected_list.cpu().numpy()

            # 生成 pred_rel_list
            pred_rel_list = [node_pairs_valid_dict[tuple(sorted((int(xy[0]), int(xy[1]))))] for xy in
                             mst_tree_selected_list if xy[0] != xy[1]]

            # 转换为 torch tensor
            pred_rel = torch.tensor(pred_rel_list).cpu().numpy()

            if map_:
                pred_edges_boxes_score.append(relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy())
                # [array([0.99997807, 0.9999968 , 0.9999999 , 0.9999969 , 0.99987507, 0.99976724], dtype=float32)]
                pred_edges_boxes_class.append(torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy())
                # [array([1, 1, 1, 1, 1, 1], dtype=int64)]
        else:
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    if map_:
        return pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score, pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class
    else:
        return pred_nodes, pred_edges

def relation_infer_gnn(h, out, model, obj_token, rln_token, nms=False, map_=False):
    # all token except the last one is object token
    # 'pred_nodes':
    # tensor([[[0.7238, 0.1392, 0.1952, 0.1962],
    #          [0.4835, 0.0991, 0.1947, 0.1953],
    #          [0.1841, 0.0691, 0.1968, 0.1965],
    #          [0.2917, 0.1657, 0.2023, 0.1995],
    #          [0.1786, 0.2498, 0.1958, 0.1939],
    #          [0.3548, 0.3452, 0.1995, 0.1996],
    #          [0.4373, 0.5390, 0.1949, 0.1942],
    #          [0.7165, 0.0849, 0.1981, 0.1984],
    #          [0.3560, 0.3794, 0.1949, 0.1948],
    #          [0.3610, 0.1147, 0.1888, 0.1893],
    #          [0.2132, 0.1397, 0.1995, 0.1987],
    #          [0.5068, 0.2446, 0.1969, 0.1979],
    #          [0.5107, 0.1138, 0.1953, 0.1966],
    #          [0.4928, 0.8826, 0.1961, 0.1970],
    #          [0.3503, 0.2831, 0.1963, 0.1963],
    #          [0.4440, 0.1837, 0.2069, 0.2058],
    #          [0.2771, 0.3296, 0.1948, 0.1947],
    #          [0.8024, 0.0798, 0.2014, 0.2011],
    #          [0.5259, 0.2376, 0.1981, 0.1975],
    #          [0.2815, 0.3279, 0.2002, 0.2009]],
    #         [[0.6836, 0.2759, 0.1979, 0.1985],
    #          [0.4160, 0.2413, 0.2074, 0.2042],
    #          [0.2138, 0.1994, 0.1946, 0.1947],
    #          [0.3712, 0.1612, 0.2030, 0.2010],
    #          [0.3974, 0.2684, 0.1943, 0.1955],
    #          [0.6089, 0.3865, 0.1902, 0.1924],
    #          [0.5396, 0.6570, 0.1961, 0.1966],
    #          [0.6644, 0.0989, 0.2000, 0.1989],
    #          [0.4631, 0.4353, 0.1952, 0.1945],
    #          [0.3708, 0.1896, 0.1978, 0.1972],
    #          [0.3530, 0.1280, 0.1955, 0.1963],
    #          [0.8104, 0.4228, 0.2088, 0.2126],
    #          [0.7645, 0.3696, 0.2109, 0.2123],
    #          [0.4950, 0.8844, 0.1958, 0.1968],
    #          [0.3818, 0.1117, 0.1964, 0.1975],
    #          [0.8104, 0.0487, 0.1946, 0.1948],
    #          [0.1969, 0.2158, 0.1977, 0.1967],
    #          [0.8167, 0.1104, 0.2093, 0.2096],
    #          [0.7700, 0.3660, 0.1953, 0.1944],
    #          [0.9047, 0.1945, 0.1928, 0.1957]]], device='cuda:0')}
    # 2 21 256    dict  model   20 1   F  True
    object_token = h[..., :obj_token, :]
    # 2 20 256
    # last token is relation token
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]
        # 2 1 256

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()
    # 返回指定维度最大值的序号
    # 第一个>第二个 0
    # 第二个>第一个 1
    # 'pred_logits':
    #  tensor([[[4.2759, -4.9492],  0
    #          [ 4.7401, -5.0336],  0
    #          [-4.2770,  4.9252],  1
    #          [ 4.6597, -4.3927],  0
    #          [ 3.0228, -2.4876],  1
    #          [ 2.3914, -2.8800],  1
    #          [-4.4479,  4.6817],  0
    #          ...
    #         [[ 3.5328, -3.9187],  0
    #          [ 2.1835, -1.8711],  0
    #          [-3.5960,  4.7048],  1
    #          [ 2.2582, -2.0884],  0
    #          [-3.5567,  4.6325],  1
    #          [ 3.8503, -4.4690],  0
    #          [-5.4784,  6.0805],  1
    #          ....
    # tensor([[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]],
    #        device='cuda:0')

    # apply nms on valid tokens
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            valid_token_id = torch.nonzero(token).squeeze(1)

            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # 0 <= x1 < x2 and 0 <= y1 < y2 has to be fulfilled
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            ids2keep = batched_nms(
                boxes=valid_nodes * 1000, scores=valid_scores, idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            # print(valid_nodes.shape[0] - ids2keep.shape[0])

            valid_token_nms[idx][valid_token_id_nms] = 1
        valid_token = valid_token_nms

    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []

        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        # batch_id 0
        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        # tensor([ 2,  6,  8, 12, 13, 16, 18], device='cuda:0')

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())
        # 前2个为坐标  后面的为0.2 0.2 所以不需要
        # [tensor([[0.1841, 0.0691],
        #         [0.4373, 0.5390],
        #         [0.3560, 0.3794],
        #         [0.5107, 0.1138],
        #         [0.4928, 0.8826],
        #         [0.2771, 0.3296],
        #         [0.5259, 0.2376]], device='cuda:0')]

        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            # [array([[0.18412243, 0.06907109, 0.1967704 , 0.19651298],
            #        [0.43725446, 0.5389896 , 0.19494587, 0.19415428],
            #        [0.3559776 , 0.3794247 , 0.19489756, 0.19481266],
            #        [0.51065665, 0.11383494, 0.19533527, 0.1966407 ],
            #        [0.49280432, 0.88256013, 0.19605483, 0.19701621],
            #        [0.2771306 , 0.32960707, 0.19484201, 0.1947407 ],
            #        [0.52590996, 0.23761915, 0.19810419, 0.19754113]], dtype=float32)]  加入了0.2 0.2
            pred_nodes_boxes_score.append(out['pred_logits'].softmax(-1)[
                                              batch_id, node_id, 1].detach().cpu().numpy())  # TODO: generalize over multi-class
            # [array([0.99989915, 0.99989164, 0.99963665, 0.99967265, 0.99997485,
            #        0.9995617 , 0.99777764], dtype=float32)]
            #        对最后一个维度取softmax 得到每一种概率 然后只取 指定的batch 指定的node_id的第二维度的值  class：选择 【0，1】不选择【1，0】
            # 所以里面的值都很接近一

            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())
            # [array([1, 1, 1, 1, 1, 1, 1], dtype=int64)]

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            n = out['pred_nodes'][batch_id, node_id, :2].detach()
            rearranged_object_token = object_token[batch_id, node_id, :]
            full_adj = torch.ones((n.shape[0], n.shape[0]), device=h.device)
            all_full_adj = []
            for row in range(full_adj.shape[0]):
                for col in range(full_adj.shape[1]):
                    all_full_adj.append([row, col])
            all_full_adj = torch.tensor(all_full_adj, device=h.device, dtype=torch.long).t().contiguous()

            # concatenate valid object pairs relation feature
            if rln_token > 0:
                relation_feature = torch.cat([rearranged_object_token,  # 6 256 + 1 256 *6
                                              relation_token[batch_id, ...].repeat(rearranged_object_token.shape[0], 1)], 1)
                val_z = model.module.GAE_model.encode(relation_feature,all_full_adj).detach()
                prob_adj = model.module.GAE_model.decoder.forward_all(val_z)
            else:
                relation_feature = rearranged_object_token
                val_z = model.module.GAE_model.encode(relation_feature, all_full_adj).detach()
                prob_adj = model.module.GAE_model.decoder.forward_all(val_z)

            prob_adj = prob_adj * torch.triu(torch.ones_like(prob_adj), diagonal=1)
            pred_rel = torch.where(prob_adj > 0.5, 1, 0)
            pred_edges.append(torch.nonzero(pred_rel).cpu().numpy())
            # [array([[0, 2],
            #        [1, 2],
            #        [1, 4],
            #        [1, 6],
            #        [2, 5],
            #        [3, 6]], dtype=int64)]

            if map_:
                ids = torch.nonzero(pred_rel)
                boxes_score_list = []
                boxes_class_list = []
                for row in range(ids.shape[0]):
                    x, y = ids[row]
                    boxes_score_list.append(prob_adj[x, y])
                    boxes_class_list.append(1)
                boxes_score_list = torch.tensor(boxes_score_list).cpu().numpy()
                boxes_class_list = torch.tensor(boxes_class_list, dtype=torch.long).cpu().numpy()
                pred_edges_boxes_score.append(boxes_score_list)
                # [array([0.99997807, 0.9999968 , 0.9999999 , 0.9999969 , 0.99987507, 0.99976724], dtype=float32)]
                pred_edges_boxes_class.append(boxes_class_list)
                # [array([1, 1, 1, 1, 1, 1], dtype=int64)]
        else:
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    if map_:
        return pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score, pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class
    else:
        return pred_nodes, pred_edges

# ########################################################################################
# ########################################################################################
# #################################  TRAIN FUNCTIONS #####################################
# ########################################################################################
# ########################################################################################
import time

def epoch_train(train_loader, net, loss_function, optimizer,
                device, last_epoch, epoch_now, max_epoch):
    # 开启 anomaly detection
    # torch.autograd.set_detect_anomaly(True)

    net.train()
    loss_all = [], [], [], [], [], []
    # check_grad = []
    all_len = len(train_loader)
    for i, batchdata in enumerate(train_loader):
        batch_start = time.time()
        # ===================get batch===================
        images, nodes, edges = batchdata[0][0], batchdata[0][1], batchdata[0][2]
        # images = images.to(device)
        images = [img.to(device) for img in images]

        nodes = [node.to(device) for node in nodes]
        edges = [edge.to(device) for edge in edges]
        target = {'nodes': nodes, 'edges': edges}
        # detr_ids = batchdata[0][-1]

        # ====================net=====================
        h, out = net(images)
        # ================compute losses=================
        losses = loss_function(h, out, target, epoch_now, max_epoch, last_epoch)
        loss_all[0].append(losses['total'].item())
        loss_all[1].append(losses['class'].item())
        loss_all[2].append(losses['nodes'].item())
        loss_all[3].append(losses['edges'].item())
        loss_all[4].append(losses['boxes'].item())
        loss_all[5].append(losses['cards'].item())



        batch_end = time.time() - batch_start
        if dist.get_rank() == 0 and i % 100 == 0:
            print(
                'Epoch: {} / {} Batch: {} / {} || Train total: {:.4f} class: {:.4f} nodes: {:.4f} edges: {:.4f} boxes: {:.4f} cards: {:.4f} take {:.4f} sec.'
                    .format(epoch_now - 1, max_epoch, i, all_len, losses['total'], losses['class'], losses['nodes'], losses['edges'], losses['boxes'], losses['cards'], batch_end))
        # ===================backward====================
        optimizer.zero_grad()
        # loss = losses['total'] + losses['class'] + losses['nodes'] + losses['edges'] + losses['boxes'] + losses['cards']
        # loss.backward()

        # with autograd.detect_anomaly():
        #     losses['total'].backward()

        # for key in losses.keys():
        #     if key != 'total':
        #         if torch.isnan(losses[key]).any():
        #             print(f"NaN detected in '{key}' loss. At {dist.get_rank()} gpu. Name is {detr_ids}")
        #             print(check_grad)
        #             break
        losses['total'].backward()

        # check_grad.append([(name, param.grad) for name, param in net.named_parameters()])
        # check_grad.pop(0)

        # check_grad.append(losses['class'].item())
        # check_grad.append(losses['nodes'].item())
        # check_grad.append(losses['edges'].item())
        # check_grad.append(losses['boxes'].item())
        # check_grad.append(losses['cards'].item())

        # for name, param in net.named_parameters():
        #     if param.grad is None and param.requires_grad is True:
        #         print(name)

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)

        # if dist.get_rank() == 0:
        #     print("/"*20)
        #     for name in net.module.state_dict():
        #         print("/"*20, name, ":", net.module.state_dict()[name])
        # torch.nn.utils.clip_grad_value_(net.parameters(),clip_value=0.1)

        # if dist.get_rank() == 0:
        #     print("S"*20)
        #     for name in net.module.state_dict():
        #         print("S"*20, name, ":", net.module.state_dict()[name])
        optimizer.step()

        # if dist.get_rank() == 0:
        #     print("P"*20)
        #     for name in net.module.state_dict():
        #         print("P"*20, name, ":", net.module.state_dict()[name])

    res_total = sum(loss_all[0]) / len(loss_all[0])
    res_class = sum(loss_all[1]) / len(loss_all[1])
    res_nodes = sum(loss_all[2]) / len(loss_all[2])
    res_edges = sum(loss_all[3]) / len(loss_all[3])
    res_boxes = sum(loss_all[4]) / len(loss_all[4])
    res_cards = sum(loss_all[5]) / len(loss_all[5])

    return res_total, res_class, res_nodes, res_edges, res_boxes, res_cards


def epoch_val(val_loader, net, config, device, SMD, args):
    net.eval()
    # if dist.get_rank() == 0:
    # print(len(val_loader))
    loss_all = []
    for i, batchdata in enumerate(val_loader):

        # ===================get batch===================
        images, nodes, edges = batchdata[0][0], batchdata[0][1], batchdata[0][2]
        # images = images.to(device)
        images = [img.to(device) for img in images]

        nodes = [node.to(device) for node in nodes]
        edges = [edge.to(device) for edge in edges]

        # ====================net=====================
        h, out = net(images)
        if args.use_gnn:
            pred_nodes, pred_edges = relation_infer_gnn(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN
            )
        else:
            if args.use_mst_train:
                pred_nodes, pred_edges = relation_infer_mst(
                    h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN
                )
            else:
                pred_nodes, pred_edges = relation_infer(
                    h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN
                )
            # pred_nodes, pred_edges = relation_infer(
            #     h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN
            # )
        # ====================compute losses=====================
        a = SMD.__call__(node_list=nodes, edge_list=edges,
                         pred_node_list=pred_nodes, pred_edge_list=pred_edges)
        smd = torch.sum(a)
        loss_all.append(smd.item())

    res = sum(loss_all) / len(loss_all)

    return res

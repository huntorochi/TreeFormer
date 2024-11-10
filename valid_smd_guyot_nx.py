import os
import yaml
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, sampler
from matplotlib import pyplot as plt
import copy
import cv2
from monai.utils import MetricReduction
import torchvision.transforms.functional as TF



# parser = ArgumentParser()
# parser.add_argument('--config',
#                     default=None,
#                     help='config file (.yml) containing the hyper-parameters for training. '
#                          'If None, use the nnU-Net config. See /config for examples.')
# parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
# parser.add_argument('--device', default='cuda',
#                         help='device to use for training')
# parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
#                         help='list of index where skip conn will be made.')
# parser.add_argument('--visualize', default=None, help='path to save visualized graphs to.')

########################################################################################################################
# detr-gnn-dataset
from skimage import exposure
import networkx as nx
import cv2
import random

def find_segments_v2(start_node, node_collections, branching_nodes, end_nodes):
    segments = []
    visited_nodes = set()

    def dfs(node, path):
        visited_nodes.add(node)
        path.append(node)

        if node in branching_nodes:
            segments.append(path.copy())
            # 继续探索分歧节点的每个邻居，从分歧节点开始新的路径
            for collection in node_collections:
                if node in collection:
                    for neighbor in collection:
                        if neighbor not in visited_nodes:
                            dfs(neighbor, [node])
            return

        if node in end_nodes:
            segments.append(path.copy())
            return

        for collection in node_collections:
            if node in collection:
                for neighbor in collection:
                    if neighbor not in visited_nodes:
                        dfs(neighbor, path.copy())

    dfs(start_node, [])

    return segments
########################################################################################################################
from math import atan2, degrees
def calculate_angle(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle = atan2(delta_y, delta_x)
    return degrees(angle) % 360

def sort_segments_v3(segments, points, branching_nodes):
    sorted_segments = []
    processed_segments = set()

    for segment in segments:
        start_node = segment[0]
        end_node = segment[-1]

        if tuple(segment) in processed_segments:
            continue

        if start_node in branching_nodes:
            related_segments = [seg for seg in segments if seg[0] == start_node and tuple(seg) not in processed_segments]

            prev_segment = [seg for seg in sorted_segments if seg[-1] == start_node]
            if prev_segment:
                prev_direction = calculate_angle(points[prev_segment[0][-2]], points[start_node])
            else:
                prev_direction = None

            related_segments.sort(
                key=lambda x: abs(calculate_angle(points[start_node], points[x[1]]) - prev_direction)
                if prev_direction is not None else 0
            )

            sorted_segments.extend(related_segments)
            processed_segments.update(tuple(seg) for seg in related_segments)
        else:
            sorted_segments.append(segment)
            processed_segments.add(tuple(segment))

    return sorted_segments
########################################################################################################################
def generate_PAFs(height, width, points, paths, line_thickness=2):
    PAFs = np.zeros((height, width, 2), dtype=np.float32)

    for branch in paths:
        for idx in range(len(branch) - 1):
            start_point = points[branch[idx]]
            end_point = points[branch[idx + 1]]
            x1, y1 = int(start_point[0] * width), int(start_point[1] * height)
            x2, y2 = int(end_point[0] * width), int(end_point[1] * height)

            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length == 0:
                continue
            ux = (x2 - x1) / length
            uy = (y2 - y1) / length

            for t in np.linspace(0, 1, int(length)):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    PAFs[y - line_thickness:y + line_thickness, x - line_thickness:x + line_thickness, 0] = ux
                    PAFs[y - line_thickness:y + line_thickness, x - line_thickness:x + line_thickness, 1] = uy

    return PAFs
##############################################################
import copy
def create_mask_with_polylines(image_shape, keypoints, segments, thickness=2):
    kpts = copy.deepcopy(keypoints)
    # Scale keypoints to match the image dimensions
    kpts[:, 0] *= image_shape[1]
    kpts[:, 1] *= image_shape[0]
    mask = np.zeros(image_shape, dtype=np.uint8)
    for segment in segments:
        segment_points = kpts[segment].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [segment_points], isClosed=False, color=1, thickness=thickness)
    return mask
##############################################################
def generate_heatmap(normalized_kpts, image_size, sigma):
    H, W = image_size
    heatmap = np.zeros((H, W))
    for keypoint in normalized_kpts:
        x_normalized, y_normalized = keypoint
        x = x_normalized * W
        y = y_normalized * H
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        gaussian = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2)
        gaussian[gaussian < 0.01] = 0
        heatmap = np.maximum(heatmap, gaussian)
    return heatmap
########################################################################################################################
# detr-gnn-dataset
def load_detr_dataset(tgt_data_path):
    path_list = []
    for file in os.listdir(tgt_data_path):
        path_list.append(file)

    list_DETR_points_left_up = []
    list_DETR_node_collections = []
    ids = path_list

    for id in ids:
        datapoint = torch.load(tgt_data_path + '/' + id)
        DETR_points_left_up = datapoint.list_DETR_points_left_up
        DETR_node_collections = datapoint.DETR_node_collections

        list_DETR_points_left_up.append(DETR_points_left_up)
        list_DETR_node_collections.append(DETR_node_collections)
    return ids, (list_DETR_points_left_up, list_DETR_node_collections)

#########################################################################################################
class LoadCNNDataset(Dataset):
    def __init__(self, parent_path, max_size=1000,
                 max_change_light_rate=0.3, is_train=True, is_rotate=False):
        self.parent_path = parent_path
        self.tgt_data_path = os.path.join(parent_path, "data")
        self.img_path = os.path.join(parent_path, "img")
        ids1 = [file for file in os.listdir(self.tgt_data_path) if file.endswith(".pt")]
        self.ids1 = ids1
        self.file_list = self.processed_file_names

        self.max_size = max_size

        self.max_change_light_rate = max_change_light_rate
        self.is_train = is_train
        self.is_rotate = is_rotate

    @property
    def processed_file_names(self):
        return self.ids1
        # path_list = []
        # for file in os.listdir(self.tgt_data_path):
        #     if file.endswith(".pt"):
        #         path_list.append(file)
        # return path_list
        # return ['data_1.pt', 'data_2.pt', ...]

    def __len__(self):
        return len(self.processed_file_names)

    def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
        """
         添加高斯噪声
        :param image: 输入的图像
        :param mu: 均值
        :param sigma: 标准差
        :return: 含有高斯噪声的图像
        """
        gasuss_img = copy.deepcopy(image)
        # image = np.array(image / 255, dtype=float)
        # gasuss_img = gasuss_img.astype(np.float32) / 255
        gasuss_img = gasuss_img.astype(np.float32)
        noise = np.random.normal(mu, sigma, gasuss_img.shape)
        gauss_noise = gasuss_img + noise
        gauss_noise = np.clip(gauss_noise, 0.0, 1.0)
        # gauss_noise = np.uint8(gauss_noise * 255)
        return gauss_noise

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        noise_img = copy.deepcopy(img)
        # return random_noise(noise_img, mode='gaussian', seed=int(time.time()), clip=True) * 255
        return self._gasuss_noise(noise_img)

    def _flip(self, img, nodes_list, nodes_list2):
        flip_nodes_list = copy.deepcopy(nodes_list)
        flip_nodes_list2 = copy.deepcopy(nodes_list2)
        flip_img = copy.deepcopy(img)
        w = flip_img.shape[1]
        # h = flip_img.shape[0]
        # if b < 0.3:

        img2 = cv2.flip(flip_img, 1)  # 参数设为1，表示左右翻转，参数设为0，则表示上下翻转
        # ---------------------- 矫正bbox坐标 ----------------------
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        flip_new_nodes_list = list()
        flip_new_nodes_list2 = list()
        for x, y in flip_nodes_list:
            flip_new_nodes_list.append([w - x, y])
        for x, y in flip_nodes_list2:
            flip_new_nodes_list2.append([w - x, y])

        return img2, flip_new_nodes_list, flip_new_nodes_list2

    def _flip2(self, img, nodes_list):
        flip_nodes_list = copy.deepcopy(nodes_list)
        flip_img = copy.deepcopy(img)
        w = flip_img.shape[1]
        # h = flip_img.shape[0]
        # if b < 0.3:

        img2 = cv2.flip(flip_img, 1)  # 参数设为1，表示左右翻转，参数设为0，则表示上下翻转
        # ---------------------- 矫正bbox坐标 ----------------------
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        flip_new_nodes_list = list()
        for x, y in flip_nodes_list:
            flip_new_nodes_list.append([w - x, y])

        return img2, flip_new_nodes_list

    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(1 - self.max_change_light_rate, 1 + self.max_change_light_rate)  # flag>1为调暗,小于1为调亮
        light_img = copy.deepcopy(img)
        return exposure.adjust_gamma(light_img, flag)

    def generate_PAFs_by_idx(self, list_DETR_points_left_up_idx, list_DETR_node_collections_idx, feature_size,
                             sigma=3, unet_thickness=2, mask_thickness=6):
        DETR_points_left_up = list_DETR_points_left_up_idx.tolist()
        DETR_node_collections = list_DETR_node_collections_idx.tolist()
        kpts = list_DETR_points_left_up_idx.numpy()

        height, width = feature_size[0], feature_size[1]
        orig_size = (height, width)  # 示例原始图像大小
        sigma = sigma  # 高斯的标准偏差
        start_node = 0

        # 构建Graph
        G = nx.Graph()
        for collection in DETR_node_collections:
            for i in range(len(collection) - 1):
                G.add_edge(collection[i], collection[i + 1])
        # 找到分歧点，即度数大于2的点
        branching_nodes = [node for node, degree in G.degree() if degree > 2]

        # 找到终点，即度数为1的点，但不包括起点
        end_nodes = [node for node, degree in G.degree() if degree == 1 and node != start_node]
        segments = find_segments_v2(start_node, DETR_node_collections, branching_nodes, end_nodes)
        sorted_segments_v3 = sort_segments_v3(segments, DETR_points_left_up, branching_nodes)
        PAFs = generate_PAFs(height, width, DETR_points_left_up, sorted_segments_v3)
        # PAFs = torch.tensor(PAFs, device=self.device)
        PAFs_tensor = torch.tensor(PAFs)
        PAFs_mask = create_mask_with_polylines(orig_size, kpts, segments, thickness=mask_thickness)
        # mask_tensor = torch.tensor(PAFs_mask, dtype=torch.bool, device=self.device)
        mask_tensor = torch.tensor(PAFs_mask, dtype=torch.bool)
        PAFs_unet = create_mask_with_polylines(orig_size, kpts, segments, thickness=unet_thickness)
        # unet_tensor = torch.tensor(PAFs_unet, dtype=torch.float32, device=self.device)
        unet_tensor = torch.tensor(PAFs_unet, dtype=torch.float32)
        # 使用之前定义的函数生成热图
        heatmap = generate_heatmap(kpts, orig_size, sigma)
        # heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32, device=self.device)
        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32)

        return PAFs_tensor, mask_tensor, unet_tensor, heatmap_tensor

    def _augment_one_sample(self, check_img, nodes_list):
        height, width, channels = check_img.shape
        a = random.random()
        if a < 0.2:
            crop_img = self._changeLight(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
        elif 0.2 <= a < 0.3:
            crop_img = self._addNoise(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)
        else:
            c = random.random()
            if c < 0.8:
                crop_img = self._changeLight(check_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
            elif 0.8 <= c < 0.9:
                crop_img = self._addNoise(check_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
            else:
                crop_img = self._changeLight(check_img)
                crop_img = self._addNoise(crop_img)
                crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)

        output_nodes = np.array(nodes_list_check)

        if crop_img.shape[0] == height and crop_img.shape[1] == width:
            output_nodes = output_nodes / np.array([width, height])
            return [1, crop_img, output_nodes, 0]
        else:
            new_height, new_width = crop_img.shape[0], crop_img.shape[1]
            output_nodes = output_nodes / np.array([new_width, new_height])
            new_img = cv2.resize(crop_img, (width, height))
            return [1, new_img, output_nodes, 0]

    def _add_edge_branch(self, img, nodes_tensor, connect_tensor, final_nodes_tensor, final_connect_tensor, M):
        def get_children(graph, node):
            out_edges = graph.out_edges(node)
            children = [target for source, target in out_edges]
            return children

        def find_node_by_point(graph, point):
            # 遍历图中的所有节点，找到与给定点匹配的节点
            for node, data in graph.nodes(data=True):
                if data['point'] == point:
                    return node
            return None  # 如果没有找到匹配的节点，返回None

        old_img = copy.deepcopy(img).cpu().numpy()
        C, height, width = old_img.shape
        edge_func = {
            'top': [0, 1, 0],
            'bottom': [0, 1, -height],
            'left': [1, 0, 0],
            'right': [1, 0, -width]
        }
        # 转换维度从 (channels, height, width) 到 (height, width, channels)

        check_nodes_list = list()
        for x, y in (nodes_tensor * torch.tensor([width, height], device=nodes_tensor.device)).cpu().numpy():
            x1 = M[0][0] * x + M[0][1] * y + M[0][2]
            y1 = M[1][0] * x + M[1][1] * y + M[1][2]
            check_nodes_list.append([int(x1), int(y1)])

        check_G = nx.DiGraph()
        rotate_G = nx.DiGraph()

        for i, point in enumerate(check_nodes_list):
            check_G.add_node(i, point=point)
        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in check_G.nodes and end in check_G.nodes:
                check_G.add_edge(start, end)

        rotate_nodes_list = [[int(x * width), int(y * height)] for x, y in final_nodes_tensor.cpu().numpy()]
        for i, point in enumerate(rotate_nodes_list):
            rotate_G.add_node(i, point=point)
        for connection in final_connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in rotate_G.nodes and end in rotate_G.nodes:
                rotate_G.add_edge(start, end)

        check_nodes_data = [check_G.nodes[node]['point'] for node in check_G.nodes]
        rotate_nodes_data = [rotate_G.nodes[node]['point'] for node in rotate_G.nodes]

        # 将列表转换为numpy数组以方便计算
        check_nodes_array = np.array(check_nodes_data)
        rotate_nodes_array = np.array(rotate_nodes_data)

        # 初始化一个空的列表来存储更新后的check节点
        updated_check_nodes_data = check_nodes_data.copy()  # 开始时复制check_nodes_data

        for rotate_node in rotate_nodes_array:
            # 计算rotate_node与check_nodes_array中所有点之间的欧氏距离
            distances = np.linalg.norm(check_nodes_array - rotate_node, axis=1)
            # 找到距离最近的check节点的索引
            nearest_idx = np.argmin(distances)
            # 使用rotate_node替换最近的check节点
            updated_check_nodes_data[nearest_idx] = list(rotate_node)

        # 首先清除check_G中的现有节点和边
        check_G.clear()
        # 根据updated_check_nodes_data重新添加节点到check_G
        for i, point in enumerate(updated_check_nodes_data):
            check_G.add_node(i, point=point)

        # 根据connect_tensor重新添加边到check_G
        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in check_G.nodes and end in check_G.nodes:
                check_G.add_edge(start, end)
        check_nodes_data = [check_G.nodes[node]['point'] for node in check_G.nodes]
        rotate_start_node = 0
        rotate_end_nodes = [node for node, degree in rotate_G.degree() if degree == 1 and node != rotate_start_node]
        # 示例：获取rotate_end_nodes中每个节点在check_G中对应节点的子节点
        for rotate_end_node in rotate_end_nodes:
            # 获取rotate_end_node的坐标
            rotate_end_point = rotate_G.nodes[rotate_end_node]['point']
            # 在check_G中找到匹配的节点
            check_node = find_node_by_point(check_G, rotate_end_point)
            if check_node is not None:
                # 如果找到了匹配的节点，获取其子节点
                check_children = get_children(check_G, check_node)
                if len(check_children) == 1:
                    check_children_point_x, check_children_point_y = check_G.nodes[check_children[0]]['point']
                    # 首先，计算点到四条边的距离
                    distance_to_edges = {
                        'top': abs(check_children_point_y),
                        'bottom': abs(height - check_children_point_y),
                        'left': abs(check_children_point_x),
                        'right': abs(width - check_children_point_x)
                    }
                    # 然后，使用 min 函数找到最短的距离及其对应的边
                    closest_edge_key, shortest_distance = min(distance_to_edges.items(), key=lambda item: item[1])
                    # 如果你需要获取最近边的方程系数，你可以从 edge_func 字典中获取
                    closest_edge_func = edge_func[closest_edge_key]
                    # 线段的两个端点的坐标
                    x1, y1 = check_G.nodes[check_children[0]]['point']
                    x2, y2 = rotate_end_point

                    # 最近边的方程系数
                    A, B, C = closest_edge_func
                    # 计算分母的值
                    denominator = A * (x2 - x1) + B * (y2 - y1)
                    # 如果分母为零，说明线段平行于最近的边，因此我们不需要做任何事情
                    if denominator != 0:
                        # 解方程以获取 t 的值
                        t = - (A * x1 + B * y1 + C) / denominator

                        # 使用 t 的值计算交点的坐标
                        intersection_x = x1 + t * (x2 - x1)
                        intersection_y = y1 + t * (y2 - y1)

                        # 使用 numpy 的 clip 函数来限制交点的坐标，确保它们在图像边界内
                        clamped_intersection_x = np.clip(int(intersection_x), 0, width)
                        clamped_intersection_y = np.clip(int(intersection_y), 0, height)

                        new_child_point = [int(clamped_intersection_x), int(clamped_intersection_y)]

                        # 计算 new_child_point 和 rotate_end_point 之间的欧几里得距离
                        distance = ((new_child_point[0] - rotate_end_point[0]) ** 2 + (
                                new_child_point[1] - rotate_end_point[1]) ** 2) ** 0.5

                        if new_child_point not in check_nodes_data and 3 < distance < 16:
                            rotate_G.add_node(len(rotate_G.nodes), point=new_child_point)
                            rotate_G.add_edge(rotate_end_node, len(rotate_G.nodes) - 1)

        rotate_nodes_data = [rotate_G.nodes[node]['point'] for node in rotate_G.nodes]
        final_rotate_nodes_tensor = torch.tensor(rotate_nodes_data, dtype=torch.float32,
                                                 device=nodes_tensor.device) / torch.tensor(
            [width, height], dtype=torch.float32, device=nodes_tensor.device)
        # 创建一个旧索引到新索引的映射
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(rotate_G.nodes)}
        # 更新边的索引
        updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in rotate_G.edges]
        # 转换为tensor
        final_rotate_connect_tensor = torch.tensor(updated_edges, dtype=torch.long, device=connect_tensor.device)

        return final_rotate_nodes_tensor, final_rotate_connect_tensor

    def _rotate(self, img, nodes_tensor, connect_tensor):
        rotate_nodes_tensor = copy.deepcopy(nodes_tensor)
        rotate_img = copy.deepcopy(img).cpu().numpy()
        C, height, width = rotate_img.shape
        # 转换维度从 (channels, height, width) 到 (height, width, channels)
        rotate_img = np.transpose(rotate_img, (1, 2, 0))
        # if b < 0.3:
        angle = random.randint(-15, 15)
        # angle = 15
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img2 = cv2.warpAffine(rotate_img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # ---------------------- 矫正bbox坐标 ----------------------
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rotate_new_nodes_list = list()
        # int_nodes_list = [(p[0] * width, [1] * height) for p in nodes[0].cpu().numpy()]
        # print(nodes[0].cpu().numpy())

        for x, y in (
                rotate_nodes_tensor * torch.tensor([width, height], device=rotate_nodes_tensor.device)).cpu().numpy():
            x1 = M[0][0] * x + M[0][1] * y + M[0][2]
            y1 = M[1][0] * x + M[1][1] * y + M[1][2]
            rotate_new_nodes_list.append([int(x1), int(y1)])

        G = nx.Graph()
        for i, point in enumerate(rotate_new_nodes_list):
            G.add_node(i, point=point)

        for connection in connect_tensor.cpu().numpy().tolist():
            start, end = connection
            if start in G.nodes and end in G.nodes:
                G.add_edge(start, end)

        # ---------------------- 删除图像外的节点 ----------------------
        for node in rotate_new_nodes_list:
            x, y = node
            if not (0 <= x < width and 0 <= y < height):  # Check if the point is within the image boundary
                index_id = rotate_new_nodes_list.index(node)
                if index_id in G.nodes:
                    G.remove_node(index_id)

        nodes_data = [G.nodes[node]['point'] for node in G.nodes]

        final_nodes_tensor = torch.tensor(nodes_data, dtype=torch.float32, device=nodes_tensor.device) / torch.tensor(
            [width, height], dtype=torch.float32, device=nodes_tensor.device)
        # 创建一个旧索引到新索引的映射
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(G.nodes)}
        # 更新边的索引
        updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in G.edges]
        # 转换为tensor
        final_connect_tensor = torch.tensor(updated_edges, dtype=torch.long, device=connect_tensor.device)
        # 将 img2 从 (height, width, channels) 转换回 (channels, height, width)
        img2 = np.transpose(img2, (2, 0, 1))
        img2_tensor = torch.tensor(img2, dtype=torch.float32, device=nodes_tensor.device)

        final_nodes_tensor, final_connect_tensor = self._add_edge_branch(img, nodes_tensor, connect_tensor,
                                                                         final_nodes_tensor,
                                                                         final_connect_tensor, M)
        return img2_tensor, final_nodes_tensor, final_connect_tensor, M

    def _rotate_tensor(self, tensor0, M, dsize):
        # 创建输入张量的一个新副本，以确保原始张量不受影响
        tensor = copy.deepcopy(tensor0)

        # 检查输入张量的维度，以确定是否需要添加通道维度
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)  # 增加通道维度
            remove_channel_dim = True
        else:
            remove_channel_dim = False

        # 将张量转换为numpy数组
        numpy_array = tensor.cpu().numpy()

        # 将数据类型转换为np.uint8（如果需要）
        numpy_array_uint8 = (numpy_array * 255).astype(np.uint8)
        # 应用旋转
        rotated_array_uint8 = cv2.warpAffine(numpy_array_uint8, M, dsize, flags=cv2.INTER_NEAREST)
        # 将数据类型转换回原始的数据类型
        rotated_array = rotated_array_uint8.astype(numpy_array.dtype) / 255
        # 将numpy数组转回为张量
        rotated_tensor = torch.tensor(rotated_array, device=tensor0.device, dtype=tensor0.dtype)

        # 如果之前添加了通道维度，现在移除它
        if remove_channel_dim:
            rotated_tensor = rotated_tensor.squeeze(-1)

        return rotated_tensor

    def __getitem__(self, idx):
        # print(idx)
        # data = torch.load(self.file_data_path + '/' + self.file_list[idx])
        label_img_name = self.file_list[idx].split(".pt")[0] + ".png"
        label_img_name0 = label_img_name.split(".")[0]
        data_name = self.ids1[idx]
        datapoint = torch.load(self.tgt_data_path + '/' + data_name)

        list_DETR_points_left_up_idx = datapoint.list_DETR_points_left_up
        list_DETR_node_collections_idx = datapoint.DETR_node_collections

        feature_img_name = label_img_name
        plt_img = plt.imread(os.path.join(self.img_path, feature_img_name)).astype(np.float32)
        if len(plt_img.shape) == 3 and plt_img.shape[2] == 4:
            plt_img = plt_img[:, :, :3]  # 最后一层为阿尔法 透明度全是1
        height, width, channels = plt_img.shape

        nodes_list = list_DETR_points_left_up_idx * torch.tensor([width, height])
        nodes_list = nodes_list.numpy()

        input_img = copy.deepcopy(plt_img)
        if self.is_train:
            result_list = self._augment_one_sample(input_img, nodes_list)
            feature_img, nodes = result_list[1], result_list[2]
            list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)
        else:
            feature_img = input_img
            list_DETR_points_left_up = list_DETR_points_left_up_idx

        if len(feature_img.shape) == 3 and feature_img.shape[2] == 3:
            transform_feature = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform_feature = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

        feature_img = transform_feature(feature_img)
        C, height, width = feature_img.shape
        cut_height = height // 2
        cut_width = width // 2
        feature_img = TF.resize(feature_img, size=[cut_height, cut_width])

        if max(cut_width, cut_height) > self.max_size:
            if cut_width > cut_height:
                scale = self.max_size / cut_width  # 小于1
                new_width = self.max_size
                new_height = int(cut_height * scale)
            else:
                scale = self.max_size / cut_height
                new_height = self.max_size
                new_width = int(cut_width * scale)
            feature_img = TF.resize(feature_img, size=[new_height, new_width])

        # 旋转
        if self.is_rotate:
            a = -9999
        else:
            a = 9999
        if a < 0:
            old_save_img = copy.deepcopy(feature_img)
            old_save_list_DETR_points_left_up = copy.deepcopy(list_DETR_points_left_up)
            old_save_list_DETR_node_collections_idx = copy.deepcopy(list_DETR_node_collections_idx)
            feature_img, list_DETR_points_left_up, list_DETR_node_collections_idx, M = self._rotate(feature_img,
                                                                                                    list_DETR_points_left_up,
                                                                                                    list_DETR_node_collections_idx)
            G_tree = nx.Graph()
            G_tree.add_edges_from(list_DETR_node_collections_idx.tolist())
            if not nx.is_tree(G_tree):
                # 不是树就不进行旋转了
                feature_img = old_save_img
                list_DETR_points_left_up = old_save_list_DETR_points_left_up
                list_DETR_node_collections_idx = old_save_list_DETR_node_collections_idx

        return (feature_img.contiguous(), label_img_name0,
                list_DETR_points_left_up, list_DETR_node_collections_idx,
                self.ids1[idx])


########################################################################################################################
def custom_collate_fn(batch):
    (feature_img, label_img_name0, list_DETR_points_left_up, list_DETR_node_collections,
     ids1) = zip(*batch)
    # images = torch.cat([item for item in feature_img], 0).contiguous()
    # 这里改为 是一个列表  每个图像被分别存进去
    images = [item.to(torch.float32) for item in feature_img]

    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]

    detr_ids = list(ids1)
    return [images, points_left_up, edges, detr_ids],


########################################################################################################################
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def ensure_format(bboxes):
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
    return bboxes


def plot_val_rel_sample(id_, path, image, points1, edges1, points2, edges2, attn_map=None, relative_coords=True):
    path = Path(path)
    H, W = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # image = (-1 *np.transpose(np.flip(image, 0), (1, 0, 2)) + 1) / 2
    # print(image)
    # print(image.shape)
    image = np.transpose(np.array(image), (1, 0, 2))  # 水平翻转
    # image = np.transpose(np.flip(image, 0), (0, 1, 2))

    # Displaying the image
    ax[0].imshow(image)
    ax[0].axis('off')

    # border_nodes = np.array([[1,1],[1,H-1],[W-1,H-1],[W-1,1]])
    border_nodes = np.array([[-1, -1], [-1, H + 1], [W + 1, H + 1], [W + 1, -1]])
    border_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    G1 = nx.Graph()
    G1.add_nodes_from(list(range(len(border_nodes))))
    coord_dict = {}
    tmp = [coord_dict.update({i: (pts[1], pts[0])}) for i, pts in enumerate(border_nodes)]
    for n, p in coord_dict.items():
        G1.nodes[n]['pos'] = p
    G1.add_edges_from(border_edges)

    pos = nx.get_node_attributes(G1, 'pos')
    nx.draw(G1, pos, ax=ax[1], node_size=1, node_color='darkgrey', edge_color='darkgrey', width=1.5, font_size=12,
            with_labels=False)
    nx.draw(G1, pos, ax=ax[2], node_size=1, node_color='darkgrey', edge_color='darkgrey', width=1.5, font_size=12,
            with_labels=False)

    G = nx.Graph()
    edges = [tuple(rel) for rel in edges1]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    # tmp = [coord_dict.update({nodes[i]: (W - W * pts[0], H - H * pts[1])}) for i, pts in enumerate(points1[nodes, :])]
    tmp = [coord_dict.update({nodes[i]: (W - W * pts[0], H - H * pts[1])}) for i, pts in enumerate(points1[nodes, :])]
    # tmp = [coord_dict.update({nodes[i]: (W - H * pts[0], H - W * pts[1])}) for i, pts in enumerate(points1[nodes, :])]
    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]['pos'] = p
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax[1], node_size=2, node_color='lightcoral', edge_color='mediumorchid', width=1, font_size=12,
            with_labels=False)

    G = nx.Graph()
    edges = [tuple(rel) for rel in edges2]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [coord_dict.update({nodes[i]: (W - W * pts[0], H - H * pts[1])}) for i, pts in enumerate(points2[nodes, :])]

    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]['pos'] = p
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax[2], node_size=2, node_color='lightcoral', edge_color='mediumorchid', width=1, font_size=12,
            with_labels=False)

    plt.savefig(path / f'{id_}.png', bbox_inches='tight')
    plt.clf()
    plt.close('all')
    # print(points2)
    # print("0"*20)


def plot_val_rel_sample_2(id_, path, image, points1, edges1, points2, edges2, attn_map=None, relative_coords=True):
    path = Path(path)
    H, W = image.shape[0], image.shape[1]
    image = (0.5 + image / 2) * 255  # (-1 1) (0 255)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_img_save = copy.deepcopy(image)
    check_img_save = copy.deepcopy(image)
    # print(output_img_save.shape)(570, 190, 3)
    # print(points1)
    # print(points2)

    output_adj = torch.zeros((points2.shape[0], points2.shape[0]))
    for num in range(edges2.shape[0]):
        row, col = edges2[num]
        output_adj[row, col] = 1
        output_adj[col, row] = 1

    for (x, y) in points2:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    for row in range(output_adj.shape[0]):
        for col in range(row, output_adj.shape[1]):
            if output_adj[row, col] == 1:
                cv2.line(img=output_img_save, pt1=(int(points2[row][0] * W), int(points2[row][1] * H)),
                         pt2=(int(points2[col][0] * W), int(points2[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    check_adj = torch.zeros((points1.shape[0], points1.shape[0]))
    for num in range(edges1.shape[0]):
        row, col = edges1[num]
        check_adj[row, col] = 1
        check_adj[col, row] = 1

    for (x, y) in points1:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    for row in range(check_adj.shape[0]):
        for col in range(row, check_adj.shape[1]):
            if check_adj[row, col] == 1:
                cv2.line(img=check_img_save, pt1=(int(points1[row][0] * W), int(points1[row][1] * H)),
                         pt2=(int(points1[col][0] * W), int(points1[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    big_img = np.concatenate([check_img_save, output_img_save], axis=1)
    # cv2.imwrite(str(path) + f"/out_{id_}.png", output_img_save)
    # cv2.imwrite(str(path) + f"/check_{id_}.png", check_img_save)
    cv2.imwrite(str(path) + f"/all_{id_}.png", big_img)


def plot_val_rel_sample_3(id_, path, image, points1, edges1, points2, edges2, type_data="normal",
                          attn_map=None, relative_coords=True):
    path = Path(path)
    H, W = image.shape[0], image.shape[1]
    image = (0.5 + image / 2) * 255  # (-1 1) (0 255)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_img_save = copy.deepcopy(image)
    check_img_save = copy.deepcopy(image)
    # print(output_img_save.shape)(570, 190, 3)
    # print(points1)
    # print(points2)

    output_adj = torch.zeros((points2.shape[0], points2.shape[0]))
    for num in range(edges2.shape[0]):
        row, col = edges2[num]
        output_adj[row, col] = 1
        output_adj[col, row] = 1



    for row in range(output_adj.shape[0]):
        for col in range(row, output_adj.shape[1]):
            if output_adj[row, col] == 1:
                cv2.line(img=output_img_save, pt1=(int(points2[row][0] * W), int(points2[row][1] * H)),
                         pt2=(int(points2[col][0] * W), int(points2[col][1] * H)),
                         color=(0, 0, 255), thickness=1)
    for (x, y) in points2:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=2, color=(0, 255, 255),
                   thickness=-1)

    check_adj = torch.zeros((points1.shape[0], points1.shape[0]))
    for num in range(edges1.shape[0]):
        row, col = edges1[num]
        check_adj[row, col] = 1
        check_adj[col, row] = 1



    for row in range(check_adj.shape[0]):
        for col in range(row, check_adj.shape[1]):
            if check_adj[row, col] == 1:
                cv2.line(img=check_img_save, pt1=(int(points1[row][0] * W), int(points1[row][1] * H)),
                         pt2=(int(points1[col][0] * W), int(points1[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    for (x, y) in points1:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=2, color=(0, 255, 255),
                   thickness=-1)

    big_img = np.concatenate([check_img_save, output_img_save], axis=1)
    cv2.imwrite(str(path) + f"/{type_data}_out_{id_}.png", output_img_save)
    cv2.imwrite(str(path) + f"/{type_data}_check_{id_}.png", check_img_save)
    cv2.imwrite(str(path) + f"/{type_data}_all_{id_}.png", big_img)


def plot_val_rel_sample_4(id_, path, image, points1, edges1, points2, edges2, points3, points4, type_data="normal",
                          attn_map=None, relative_coords=True):
    '''

    :param image:
    :param points1: pred
    :param edges1: pred
    :param points2: real
    :param edges2: real
    :param points3: out pred
    :param points4: out real
    :param attn_map:
    :param relative_coords:
    :return:
    '''
    path = Path(path)
    H, W = image.shape[0], image.shape[1]
    image = (0.5 + image / 2) * 255  # (-1 1) (0 255)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_img_save = copy.deepcopy(image)
    check_img_save = copy.deepcopy(image)

    output_adj = torch.zeros((points2.shape[0], points2.shape[0]))
    for num in range(edges2.shape[0]):
        row, col = edges2[num]
        output_adj[row, col] = 1
        output_adj[col, row] = 1

    # for (x, y) in points2:
    #     cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
    #                thickness=1)

    for row in range(output_adj.shape[0]):
        for col in range(row, output_adj.shape[1]):
            if output_adj[row, col] == 1:
                cv2.line(img=output_img_save, pt1=(int(points2[row][0] * W), int(points2[row][1] * H)),
                         pt2=(int(points2[col][0] * W), int(points2[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    for (x, y) in points2:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=-1)

    for (x, y) in points3:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=2, color=(255, 255, 0),
                   thickness=-1)

    check_adj = torch.zeros((points1.shape[0], points1.shape[0]))
    for num in range(edges1.shape[0]):
        row, col = edges1[num]
        check_adj[row, col] = 1
        check_adj[col, row] = 1

    # for (x, y) in points1:
    #     cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
    #                thickness=1)

    for row in range(check_adj.shape[0]):
        for col in range(row, check_adj.shape[1]):
            if check_adj[row, col] == 1:
                cv2.line(img=check_img_save, pt1=(int(points1[row][0] * W), int(points1[row][1] * H)),
                         pt2=(int(points1[col][0] * W), int(points1[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    for (x, y) in points1:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=-1)

    for (x, y) in points4:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=2, color=(255, 255, 0),
                   thickness=-1)

    big_img = np.concatenate([check_img_save, output_img_save], axis=1)
    cv2.imwrite(str(path) + f"/{type_data}_out_{id_}.png", output_img_save)
    cv2.imwrite(str(path) + f"/{type_data}_check_{id_}.png", check_img_save)
    cv2.imwrite(str(path) + f"/{type_data}_all_{id_}.png", big_img)


def plot_val_rel_sample_show(image, points1, edges1, points2, edges2, attn_map=None, relative_coords=True):
    H, W = image.shape[0], image.shape[1]
    image = (0.5 + image / 2) * 255  # (-1 1) (0 255)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_img_save = copy.deepcopy(image)
    check_img_save = copy.deepcopy(image)

    output_adj = torch.zeros((points2.shape[0], points2.shape[0]))
    for num in range(edges2.shape[0]):
        row, col = edges2[num]
        output_adj[row, col] = 1
        output_adj[col, row] = 1

    for (x, y) in points2:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    for row in range(output_adj.shape[0]):
        for col in range(row, output_adj.shape[1]):
            if output_adj[row, col] == 1:
                cv2.line(img=output_img_save, pt1=(int(points2[row][0] * W), int(points2[row][1] * H)),
                         pt2=(int(points2[col][0] * W), int(points2[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    check_adj = torch.zeros((points1.shape[0], points1.shape[0]))
    for num in range(edges1.shape[0]):
        row, col = edges1[num]
        check_adj[row, col] = 1
        check_adj[col, row] = 1

    for (x, y) in points1:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    for row in range(check_adj.shape[0]):
        for col in range(row, check_adj.shape[1]):
            if check_adj[row, col] == 1:
                cv2.line(img=check_img_save, pt1=(int(points1[row][0] * W), int(points1[row][1] * H)),
                         pt2=(int(points1[col][0] * W), int(points1[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    big_img = np.concatenate([check_img_save, output_img_save], axis=1)
    cv2.imshow("big_img", big_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_val_rel_sample_show2(image, points1, edges1, points2, edges2, points3, points4, attn_map=None, relative_coords=True):
    '''

    :param image:
    :param points1: pred
    :param edges1: pred
    :param points2: real
    :param edges2: real
    :param points3: out pred
    :param points4: out real
    :param attn_map:
    :param relative_coords:
    :return:
    '''
    H, W = image.shape[0], image.shape[1]
    image = (0.5 + image / 2) * 255  # (-1 1) (0 255)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_img_save = copy.deepcopy(image)
    check_img_save = copy.deepcopy(image)

    output_adj = torch.zeros((points2.shape[0], points2.shape[0]))
    for num in range(edges2.shape[0]):
        row, col = edges2[num]
        output_adj[row, col] = 1
        output_adj[col, row] = 1

    # for (x, y) in points2:
    #     cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
    #                thickness=1)

    for row in range(output_adj.shape[0]):
        for col in range(row, output_adj.shape[1]):
            if output_adj[row, col] == 1:
                cv2.line(img=output_img_save, pt1=(int(points2[row][0] * W), int(points2[row][1] * H)),
                         pt2=(int(points2[col][0] * W), int(points2[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    for (x, y) in points3:
        cv2.circle(img=output_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    check_adj = torch.zeros((points1.shape[0], points1.shape[0]))
    for num in range(edges1.shape[0]):
        row, col = edges1[num]
        check_adj[row, col] = 1
        check_adj[col, row] = 1

    # for (x, y) in points1:
    #     cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
    #                thickness=1)

    for row in range(check_adj.shape[0]):
        for col in range(row, check_adj.shape[1]):
            if check_adj[row, col] == 1:
                cv2.line(img=check_img_save, pt1=(int(points1[row][0] * W), int(points1[row][1] * H)),
                         pt2=(int(points1[col][0] * W), int(points1[col][1] * H)),
                         color=(0, 0, 255), thickness=1)

    for (x, y) in points4:
        cv2.circle(img=check_img_save, center=(int(x * W), int(y * H)), radius=1, color=(0, 255, 255),
                   thickness=1)

    big_img = np.concatenate([check_img_save, output_img_save], axis=1)
    cv2.imshow("big_img", big_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def full_node_graph_2_keypoint_node_graph(nodes, edges, pred_nodes, pred_edges, pred_nodes_box,
                                          pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score,
                                          pred_edges_box_class):
    # 生成图 并且删除不满足条件的节点和边
    out_nodes, out_edges, out_pred_nodes, out_pred_edges, out_pred_nodes_box = [], [], [], [], []
    out_pred_nodes_box_score, out_pred_nodes_box_class, out_pred_edges_box_score = [], [], []
    out_pred_edges_box_class = []

    for batch_num in range(len(nodes)):

        # print(nodes[0].shape,  # tensor
        #       edges[0].shape,  # tensor
        #       pred_nodes[0].shape,  # tensor
        #       pred_edges[0].shape,  # numpy.ndarray
        #       pred_nodes_box[0].shape,  # numpy.ndarray
        #       pred_nodes_box_score[0].shape,  # numpy.ndarray
        #       pred_nodes_box_class[0].shape,  # numpy.ndarray
        #       pred_edges_box_score[0].shape,  # numpy.ndarray
        #       pred_edges_box_class[0].shape)  # numpy.ndarray
        # torch.Size([16, 2]) torch.Size([15, 2]) torch.Size([19, 2]) (18, 2) (19, 4) (19,) (19,) (18,) (18,)

        #########################################################################################################
        # 首先处理pred
        test_pred_graph = nx.Graph()
        for i, no in enumerate(pred_nodes[batch_num]):
            no_num = i
            test_pred_graph.add_node(no_num, node_box=pred_nodes_box[batch_num][i],
                                     node_score=pred_nodes_box_score[batch_num][i],
                                     node_class=pred_nodes_box_class[batch_num][i],
                                     node=no.cpu().detach().numpy())
        for i, ed in enumerate(pred_edges[batch_num]):
            test_pred_graph.add_edge(ed[0], ed[1], edge_score=pred_edges_box_score[batch_num][i],
                                     edge_class=pred_edges_box_class[batch_num][i])

        # 生成节点位置
        # pos = dict([(i, nodes_pos.cpu().numpy()) for i, nodes_pos in enumerate(pred_nodes[0])])
        # print('position of all nodes:', pos)

        pred_degree_list = []
        for i in range(len(pred_nodes[batch_num])):
            if test_pred_graph.degree(i) == 2:
                pred_degree_list.append(i)
        # print(list(test_pred_graph.edges(data=True)))
        # print(list(test_pred_graph.edges()))
        # print(pred_degree_list)

        for skip_node in pred_degree_list:
            if test_pred_graph.degree(skip_node) == 2:
                edge_data_list = list(test_pred_graph.edges(skip_node, data=True))
                # print("***********************")
                # print(skip_node)
                # print(edge_data_list)
                new_edge_score = (edge_data_list[0][2]['edge_score'] + edge_data_list[1][2]['edge_score']) / 2
                new_edge_class = (edge_data_list[0][2]['edge_class'] + edge_data_list[1][2]['edge_class'] + 0.5) / 2
                # print(new_edge_class)
                neighbors_list = [n for n in test_pred_graph.neighbors(skip_node)]
                test_pred_graph.remove_node(skip_node)
                test_pred_graph.add_edge(neighbors_list[0], neighbors_list[1], edge_score=np.float32(new_edge_score),
                                         edge_class=np.int32(new_edge_class))

        # connect_list = list(set(test_pred_graph.nodes) - set(pred_degree_list))
        # print(connect_list)

        # # 把节点画出来
        # nx.draw_networkx_nodes(test_pred_graph, pos, node_color='g', node_size=500, alpha=0.8)
        #
        # # 把边画出来
        # nx.draw_networkx_edges(test_pred_graph, pos, width=1.0, alpha=0.5, edge_color='b')
        #
        # plt.axis('on')
        # # 去掉坐标刻度
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

        # print(test_pred_graph.nodes)
        # print(test_pred_graph.nodes(data=True))
        # print(type(test_pred_graph.nodes))
        # print(test_pred_graph.edges)
        # print(test_pred_graph.edges(data=True))
        # print(type(test_pred_graph.edges))
        networkx_pred_nodes = list(test_pred_graph.nodes(data=True))
        networkx_pred_edges = list(test_pred_graph.edges(data=True))
        nodes_dict = {}
        for i, data in enumerate(networkx_pred_nodes):
            nodes_dict[data[0]] = i

        batch_pred_nodes_tensor = torch.tensor([i[1]['node'] for i in networkx_pred_nodes]).to(
            pred_nodes[batch_num].device)
        if (batch_pred_nodes_tensor.dim() != 0 and batch_pred_nodes_tensor.nelement() != 0
                and batch_pred_nodes_tensor.shape[0] > 1 and len(networkx_pred_edges) > 0):
            out_pred_nodes.append(batch_pred_nodes_tensor)

            batch_pred_edges_array = np.array([[nodes_dict[i[0]], nodes_dict[i[1]]] for i in networkx_pred_edges],
                                              dtype=np.int64)
            out_pred_edges.append(batch_pred_edges_array)

            batch_pred_nodes_box_array = np.array([i[1]['node_box'] for i in networkx_pred_nodes])
            out_pred_nodes_box.append(batch_pred_nodes_box_array)

            batch_pred_nodes_box_score_array = np.array([i[1]['node_score'] for i in networkx_pred_nodes])
            out_pred_nodes_box_score.append(batch_pred_nodes_box_score_array)

            batch_pred_nodes_box_class_array = np.array([i[1]['node_class'] for i in networkx_pred_nodes])
            out_pred_nodes_box_class.append(batch_pred_nodes_box_class_array)

            batch_pred_edges_box_score_array = np.array([i[2]['edge_score'] for i in networkx_pred_edges])
            out_pred_edges_box_score.append(batch_pred_edges_box_score_array)

            batch_pred_edges_box_class_array = np.array([i[2]['edge_class'] for i in networkx_pred_edges],
                                                        dtype=np.int64)
            out_pred_edges_box_class.append(batch_pred_edges_box_class_array)
        else:
            out_pred_nodes.append(torch.empty(0, 2).to(pred_nodes[batch_num].device))
            out_pred_edges.append(torch.empty(0, 2).cpu())
            out_pred_nodes_box.append(torch.empty(0, 4).cpu().numpy())
            out_pred_nodes_box_score.append(torch.empty(0, 1).cpu().numpy())
            out_pred_nodes_box_class.append(torch.empty(0, 1).cpu().numpy())
            out_pred_edges_box_score.append(torch.empty(0, 1).cpu().numpy())
            out_pred_edges_box_class.append(torch.empty(0, 1).cpu().numpy())

        #########################################################################################################
        # 处理real
        test_real_graph = nx.Graph()
        for i, no in enumerate(nodes[batch_num]):
            no_num = i
            test_real_graph.add_node(no_num, node=no.cpu().detach().numpy())
        for i, ed in enumerate(edges[batch_num]):
            a = int(ed[0].cpu().detach().numpy())
            b = int(ed[1].cpu().detach().numpy())
            test_real_graph.add_edge(a, b)

        real_degree_list = []
        for i in range(len(nodes[batch_num])):
            if test_real_graph.degree(i) == 2:
                real_degree_list.append(i)

        for skip_node in real_degree_list:
            neighbors_list = [n for n in test_real_graph.neighbors(skip_node)]
            test_real_graph.remove_node(skip_node)
            test_real_graph.add_edge(neighbors_list[0], neighbors_list[1])

        networkx_real_nodes = list(test_real_graph.nodes(data=True))
        networkx_real_edges = list(test_real_graph.edges(data=True))
        nodes_dict = {}
        for i, data in enumerate(networkx_real_nodes):
            nodes_dict[data[0]] = i

        batch_real_nodes_tensor = torch.tensor([i[1]['node'] for i in networkx_real_nodes]).to(
            nodes[batch_num].device)
        out_nodes.append(batch_real_nodes_tensor)

        batch_real_edges_tensor = torch.tensor([[nodes_dict[i[0]], nodes_dict[i[1]]] for i in networkx_real_edges]).to(
            edges[batch_num].device)
        out_edges.append(batch_real_edges_tensor)

    return out_nodes, out_edges, out_pred_nodes, out_pred_edges, out_pred_nodes_box, out_pred_nodes_box_score, \
           out_pred_nodes_box_class, out_pred_edges_box_score, out_pred_edges_box_class,


def test(is_use_mst, args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))



    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    from models import build_model

    if is_use_mst:
        # from inference_infinity_mst_nx_dist import relation_infer
        from inference_infinity_mst_nx_gradmst import relation_infer
    else:
        from inference_infinity_gradmst import relation_infer

    from metric_smd import StreetMoverDistance
    from metric_map import BBoxEvaluator
    from metric_topo.topo import compute_topo
    from box_ops_2D import box_cxcywh_to_xyxy_np
    import random

    # num_gpus = torch.cuda.device_count()
    # print(f"Number of available GPUs: {num_gpus}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:1") if args.device == 'cuda' else torch.device("cpu")
    # device = torch.device("cpu")

    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    net = build_model(config, args=args).to(device)


    # test_path = "H:\\3D2cut_Single_Guyot\\3D2cut_Single_Guyot\\guyot_data\\test"
    # val_path = r"I:\3D2cut_Single_Guyot\all_same_PAF_move\val_aug"
    val_path = r"I:\3D2cut_Single_Guyot\all_same_PAF_move\test"
    #img_test_dataset_path1 = "F:\\new_root_dataset\\individuals_net\\final\\test\\parts\\test50/img"
    #img_test_dataset_path2 = "F:\\new_root_dataset\\individuals_net\\final\\test\\parts\\test50/unet"
    #tgt_test_dataset_path = "F:\\new_root_dataset\\individuals_net\\final\\test\\parts\\test50/data"

    dataset_val = LoadCNNDataset(parent_path=val_path, max_size=2048, max_change_light_rate=0.3, is_train=False, is_rotate=False)

    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False,
                             collate_fn=custom_collate_fn, drop_last=False, pin_memory=False,
                             num_workers=0)


    # load checkpoint
    # original saved file with DataParallel
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["net"].items():
        name = k[7:]  # remove `module.`
        # print(k, name)
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    # net.load_state_dict(checkpoint['net'])
    net.eval()

    # init metric
    # metric_smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    metric_smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    smd_results = []

    metric_smd_keypoint = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    smd_results_keypoint = []

    metric_node_map = BBoxEvaluator(['node'], max_detections=256)
    metric_edge_map = BBoxEvaluator(['edge'], max_detections=256)

    metric_node_map_keypoint = BBoxEvaluator(['node_keypoint'], max_detections=256)
    metric_edge_map_keypoint = BBoxEvaluator(['edge_keypoint'], max_detections=256)

    topo_results = []
    topo_results_keypoint = []

    all_num = 0
    tree_num = 0

    if args.visualize:
        os.makedirs(args.visualize, exist_ok=True)

    with torch.no_grad():
        print('Started processing test set.')
        for id_, batchdata in enumerate(tqdm(val_loader)):

            # extract data and put to device
            images, nodes, edges = batchdata[0][0], batchdata[0][1], batchdata[0][2]
            # images = images.to(args.device, non_blocking=False)  # 2 3 512 512
            images = [img.to(device, non_blocking=False) for img in images]  # 2 3 512 512
            nodes = [node.to(device, non_blocking=False) for node in nodes]  # 7 2    10 2
            edges = [edge.to(device, non_blocking=False) for edge in edges]  # 6 2    9 2
            detr_ids = batchdata[0][-1]

            # print('detr_ids', detr_ids)

            h, out = net(images)
            # h bs 21 256
            # out {'pred_logits': class_prob bs 20 2, 'pred_nodes': coord_loc bs 20 4}

            pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True
            )
            # 保存图片
            # Save visualization
            if args.visualize:
                save_id = detr_ids[0].split(".pt")[0]
                plot_val_rel_sample_3(
                    save_id, args.visualize,
                    images[0].permute(1, 2, 0).cpu().numpy(),
                    nodes[0].cpu().numpy(), edges[0].cpu().numpy(),
                    pred_nodes[0].cpu().numpy(), pred_edges[0], type_data="normal"
                )
            for edge in pred_edges:
                all_num += 1
                edge = edge.tolist()
                G = nx.Graph()
                G.add_edges_from(edge)
                if edge:
                    if nx.is_tree(G):
                        tree_num += 1


            out_nodes, out_edges, out_pred_nodes, out_pred_edges, out_pred_nodes_box, out_pred_nodes_box_score, \
            out_pred_nodes_box_class, out_pred_edges_box_score, \
            out_pred_edges_box_class = full_node_graph_2_keypoint_node_graph(
                nodes=nodes, edges=edges, pred_nodes=pred_nodes, pred_edges=pred_edges,
                pred_nodes_box=pred_nodes_box,
                pred_nodes_box_score=pred_nodes_box_score,
                pred_nodes_box_class=pred_nodes_box_class,
                pred_edges_box_score=pred_edges_box_score,
                pred_edges_box_class=pred_edges_box_class)

            # # 保存图片
            # Save visualization
            if args.visualize:
                save_id = detr_ids[0].split(".pt")[0]
                plot_val_rel_sample_4(
                    save_id, args.visualize,
                    images[0].permute(1, 2, 0).cpu().numpy(),
                    nodes[0].cpu().numpy(), edges[0].cpu().numpy(),
                    pred_nodes[0].cpu().numpy(), pred_edges[0],
                    out_pred_nodes[0].cpu().numpy(), out_nodes[0].cpu().numpy(), type_data="keypoint"
                )


            # Add smd of current batch elem
            # print('ret SMD')
            ret = metric_smd(nodes, edges, pred_nodes, pred_edges)  # tensor([3.3033e-05, 3.3636e-05])
            smd_results += ret.tolist()

            # 这个就是smd 不需要变成关键节点
            # 虽然这么说  但是还是改了
            # print('ret SMD_keypoint')
            ret_keypoint = metric_smd_keypoint(out_nodes, out_edges, out_pred_nodes, out_pred_edges)
            smd_results_keypoint += ret_keypoint.tolist()

            # Add elements of current batch elem to node map evaluator
            metric_node_map.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(box) for box in pred_nodes_box],
                pred_classes=pred_nodes_box_class,
                pred_scores=pred_nodes_box_score,
                gt_boxes=[
                    box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for
                    nodes_ in nodes],
                gt_classes=[np.ones((nodes_.shape[0],)) for nodes_ in nodes]
            )

            metric_node_map_keypoint.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(box) for box in out_pred_nodes_box],
                pred_classes=out_pred_nodes_box_class,
                pred_scores=out_pred_nodes_box_score,
                gt_boxes=[
                    box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for
                    nodes_ in out_nodes],
                gt_classes=[np.ones((nodes_.shape[0],)) for nodes_ in out_nodes]
            )

            # Add elements of current batch elem to edge map evaluator
            pred_edges_box = []
            out_pred_edges_box = []
            for edges_, nodes_ in zip(pred_edges, pred_nodes):
                nodes_ = nodes_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                pred_edges_box.append(edges_box)

            for edges_, nodes_ in zip(out_pred_edges, out_pred_nodes):
                nodes_ = nodes_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                out_pred_edges_box.append(edges_box)

            gt_edges_box = []
            out_gt_edges_box = []
            for edges_, nodes_ in zip(edges, nodes):
                nodes_, edges_ = nodes_.cpu().numpy(), edges_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                gt_edges_box.append(edges_box)

            for edges_, nodes_ in zip(out_edges, out_nodes):
                nodes_, edges_ = nodes_.cpu().numpy(), edges_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                out_gt_edges_box.append(edges_box)

            # if detr_ids[0] == '37.pt':
            metric_edge_map.add(
                pred_boxes=pred_edges_box,
                pred_classes=pred_edges_box_class,
                pred_scores=pred_edges_box_score,
                gt_boxes=gt_edges_box,
                gt_classes=[np.ones((edges_.shape[0],)) for edges_ in edges]
            )

            metric_edge_map_keypoint.add(
                pred_boxes=out_pred_edges_box,
                pred_classes=out_pred_edges_box_class,
                pred_scores=out_pred_edges_box_score,
                gt_boxes=out_gt_edges_box,
                gt_classes=[np.ones((edges_.shape[0],)) for edges_ in out_edges]
            )

            for node_, edge_, pred_node_, pred_edge_ in zip(nodes, edges, pred_nodes, pred_edges):
                topo_results.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_, img_size=512))

            for node_, edge_, pred_node_, pred_edge_ in zip(out_nodes, out_edges, out_pred_nodes, out_pred_edges):
                topo_results_keypoint.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_, img_size=512))

    # Determine smd
    smd_mean = torch.tensor(smd_results).mean().item()
    smd_std = torch.tensor(smd_results).std().item()

    smd_mean_keypoint = torch.tensor(smd_results_keypoint).mean().item()
    smd_std_keypoint = torch.tensor(smd_results_keypoint).std().item()
    # print(len(smd_results))
    print(f'smd value: mean {smd_mean}, std {smd_std}\n')
    print(f'smd_keypoint value_keypoint: mean {smd_mean_keypoint}, std {smd_std_keypoint}\n')

    # Save visualization
    if args.visualize:
        txt_path_checkpoint = Path(args.visualize + "/wish.txt")
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f'smd value: mean {smd_mean}, std {smd_std}\n\n')
            f.write(f'smd_keypoint value_keypoint: mean {smd_mean_keypoint}, std {smd_std_keypoint}\n')

    if args.visualize:
        detail_txt_path_checkpoint = Path(args.visualize + "/detail.txt")
        with open(detail_txt_path_checkpoint, mode="a") as d:
            d.write(f'smd value: mean {smd_mean}, std {smd_std}\n\n')
            d.write(f'smd_keypoint value_keypoint: mean {smd_mean_keypoint}, std {smd_std_keypoint}\n')
    # Determine node box ap / ar
    node_metric_scores = metric_node_map.eval()
    print(f"node mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node AP_IoU_0.10_MaxDet_100 {node_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"node AP_IoU_0.20_MaxDet_100 {node_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"node AP_IoU_0.30_MaxDet_100 {node_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"node AP_IoU_0.40_MaxDet_100 {node_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"node AP_IoU_0.50_MaxDet_100 {node_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"node AP_IoU_0.60_MaxDet_100 {node_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"node AP_IoU_0.70_MaxDet_100 {node_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"node AP_IoU_0.80_MaxDet_100 {node_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"node AP_IoU_0.90_MaxDet_100 {node_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    node_metric_scores_keypoint = metric_node_map_keypoint.eval()
    print(f"node_keypoint mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores_keypoint['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.10_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.10_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.20_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.20_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.30_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.30_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.40_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.40_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.50_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.50_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.60_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.60_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.70_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.70_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.80_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.80_MaxDet_100']}")
    print(f"node_keypoint AP_IoU_0.90_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.90_MaxDet_100']}\n")
    # Save visualization
    if args.visualize:
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"node mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"node AP_IoU_0.10_MaxDet_100 {node_metric_scores['AP_IoU_0.10_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.20_MaxDet_100 {node_metric_scores['AP_IoU_0.20_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.30_MaxDet_100 {node_metric_scores['AP_IoU_0.30_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.40_MaxDet_100 {node_metric_scores['AP_IoU_0.40_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.50_MaxDet_100 {node_metric_scores['AP_IoU_0.50_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.60_MaxDet_100 {node_metric_scores['AP_IoU_0.60_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.70_MaxDet_100 {node_metric_scores['AP_IoU_0.70_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.80_MaxDet_100 {node_metric_scores['AP_IoU_0.80_MaxDet_100']}\n")
            f.write(f"node AP_IoU_0.90_MaxDet_100 {node_metric_scores['AP_IoU_0.90_MaxDet_100']}\n\n")
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(
                f"node_keypoint mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores_keypoint['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"node_keypoint AP_IoU_0.10_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.10_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.20_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.20_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.30_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.30_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.40_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.40_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.50_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.50_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.60_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.60_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.70_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.70_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.80_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.80_MaxDet_100']}\n")
            f.write(f"node_keypoint AP_IoU_0.90_MaxDet_100 {node_metric_scores_keypoint['AP_IoU_0.90_MaxDet_100']}\n\n")

    print(f"node mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node AR_IoU_0.10_MaxDet_100 {node_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"node AR_IoU_0.20_MaxDet_100 {node_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"node AR_IoU_0.30_MaxDet_100 {node_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"node AR_IoU_0.40_MaxDet_100 {node_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"node AR_IoU_0.50_MaxDet_100 {node_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"node AR_IoU_0.60_MaxDet_100 {node_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"node AR_IoU_0.70_MaxDet_100 {node_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"node AR_IoU_0.80_MaxDet_100 {node_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"node AR_IoU_0.90_MaxDet_100 {node_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

    print(f"node_keypoint mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores_keypoint['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.10_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.10_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.20_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.20_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.30_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.30_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.40_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.40_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.50_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.50_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.60_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.60_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.70_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.70_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.80_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.80_MaxDet_100']}")
    print(f"node_keypoint AR_IoU_0.90_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.90_MaxDet_100']}\n")
    # Save visualization
    if args.visualize:
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"node mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"node AR_IoU_0.10_MaxDet_100 {node_metric_scores['AR_IoU_0.10_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.20_MaxDet_100 {node_metric_scores['AR_IoU_0.20_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.30_MaxDet_100 {node_metric_scores['AR_IoU_0.30_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.40_MaxDet_100 {node_metric_scores['AR_IoU_0.40_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.50_MaxDet_100 {node_metric_scores['AR_IoU_0.50_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.60_MaxDet_100 {node_metric_scores['AR_IoU_0.60_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.70_MaxDet_100 {node_metric_scores['AR_IoU_0.70_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.80_MaxDet_100 {node_metric_scores['AR_IoU_0.80_MaxDet_100']}\n")
            f.write(f"node AR_IoU_0.90_MaxDet_100 {node_metric_scores['AR_IoU_0.90_MaxDet_100']}\n\n")
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(
                f"node_keypoint mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores_keypoint['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"node_keypoint AR_IoU_0.10_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.10_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.20_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.20_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.30_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.30_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.40_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.40_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.50_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.50_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.60_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.60_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.70_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.70_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.80_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.80_MaxDet_100']}\n")
            f.write(f"node_keypoint AR_IoU_0.90_MaxDet_100 {node_metric_scores_keypoint['AR_IoU_0.90_MaxDet_100']}\n\n")

    # Determine edge box ap / ar
    edge_metric_scores = metric_edge_map.eval()
    print(f"edge mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge AP_IoU_0.10_MaxDet_100 {edge_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"edge AP_IoU_0.20_MaxDet_100 {edge_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"edge AP_IoU_0.30_MaxDet_100 {edge_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"edge AP_IoU_0.40_MaxDet_100 {edge_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"edge AP_IoU_0.50_MaxDet_100 {edge_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"edge AP_IoU_0.60_MaxDet_100 {edge_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"edge AP_IoU_0.70_MaxDet_100 {edge_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"edge AP_IoU_0.80_MaxDet_100 {edge_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"edge AP_IoU_0.90_MaxDet_100 {edge_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    edge_metric_scores_keypoint = metric_edge_map_keypoint.eval()
    print(f"edge_keypoint mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores_keypoint['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.10_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.10_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.20_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.20_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.30_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.30_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.40_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.40_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.50_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.50_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.60_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.60_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.70_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.70_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.80_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.80_MaxDet_100']}")
    print(f"edge_keypoint AP_IoU_0.90_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.90_MaxDet_100']}\n")
    # Save visualization
    if args.visualize:
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"edge mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"edge AP_IoU_0.10_MaxDet_100 {edge_metric_scores['AP_IoU_0.10_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.20_MaxDet_100 {edge_metric_scores['AP_IoU_0.20_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.30_MaxDet_100 {edge_metric_scores['AP_IoU_0.30_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.40_MaxDet_100 {edge_metric_scores['AP_IoU_0.40_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.50_MaxDet_100 {edge_metric_scores['AP_IoU_0.50_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.60_MaxDet_100 {edge_metric_scores['AP_IoU_0.60_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.70_MaxDet_100 {edge_metric_scores['AP_IoU_0.70_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.80_MaxDet_100 {edge_metric_scores['AP_IoU_0.80_MaxDet_100']}\n")
            f.write(f"edge AP_IoU_0.90_MaxDet_100 {edge_metric_scores['AP_IoU_0.90_MaxDet_100']}\n\n")
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(
                f"edge_keypoint mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores_keypoint['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"edge_keypoint AP_IoU_0.10_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.10_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.20_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.20_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.30_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.30_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.40_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.40_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.50_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.50_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.60_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.60_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.70_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.70_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.80_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.80_MaxDet_100']}\n")
            f.write(f"edge_keypoint AP_IoU_0.90_MaxDet_100 {edge_metric_scores_keypoint['AP_IoU_0.90_MaxDet_100']}\n\n")

    print(f"edge mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge AR_IoU_0.10_MaxDet_100 {edge_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"edge AR_IoU_0.20_MaxDet_100 {edge_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"edge AR_IoU_0.30_MaxDet_100 {edge_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"edge AR_IoU_0.40_MaxDet_100 {edge_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"edge AR_IoU_0.50_MaxDet_100 {edge_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"edge AR_IoU_0.60_MaxDet_100 {edge_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"edge AR_IoU_0.70_MaxDet_100 {edge_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"edge AR_IoU_0.80_MaxDet_100 {edge_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"edge AR_IoU_0.90_MaxDet_100 {edge_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

    print(f"edge_keypoint mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores_keypoint['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.10_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.10_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.20_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.20_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.30_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.30_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.40_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.40_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.50_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.50_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.60_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.60_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.70_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.70_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.80_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.80_MaxDet_100']}")
    print(f"edge_keypoint AR_IoU_0.90_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.90_MaxDet_100']}\n")
    # Determine topo
    P = np.array(topo_results).mean(axis=0)[0]
    R = np.array(topo_results).mean(axis=0)[1]
    F1 = (2 * P * R) / (P + R)
    print(np.array(topo_results).mean(axis=0))

    P_keypoint = np.array(topo_results_keypoint).mean(axis=0)[0]
    R_keypoint = np.array(topo_results_keypoint).mean(axis=0)[1]
    F1_keypoint = (2 * P_keypoint * R_keypoint) / (P_keypoint + R_keypoint)
    print(np.array(topo_results_keypoint).mean(axis=0))

    print(f"all num: {all_num}, tree num: {tree_num}")

    # Save visualization
    if args.visualize:
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"edge mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"edge AR_IoU_0.10_MaxDet_100 {edge_metric_scores['AR_IoU_0.10_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.20_MaxDet_100 {edge_metric_scores['AR_IoU_0.20_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.30_MaxDet_100 {edge_metric_scores['AR_IoU_0.30_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.40_MaxDet_100 {edge_metric_scores['AR_IoU_0.40_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.50_MaxDet_100 {edge_metric_scores['AR_IoU_0.50_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.60_MaxDet_100 {edge_metric_scores['AR_IoU_0.60_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.70_MaxDet_100 {edge_metric_scores['AR_IoU_0.70_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.80_MaxDet_100 {edge_metric_scores['AR_IoU_0.80_MaxDet_100']}\n")
            f.write(f"edge AR_IoU_0.90_MaxDet_100 {edge_metric_scores['AR_IoU_0.90_MaxDet_100']}\n\n")
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(
                f"edge_keypoint mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores_keypoint['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
            f.write(f"edge_keypoint AR_IoU_0.10_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.10_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.20_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.20_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.30_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.30_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.40_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.40_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.50_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.50_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.60_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.60_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.70_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.70_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.80_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.80_MaxDet_100']}\n")
            f.write(f"edge_keypoint AR_IoU_0.90_MaxDet_100 {edge_metric_scores_keypoint['AR_IoU_0.90_MaxDet_100']}\n\n")

        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"P R F1:  {np.array(topo_results).mean(axis=0)}, {F1}\n\n")

        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"P_keypoint R_keypoint F1_keypoint:  {np.array(topo_results_keypoint).mean(axis=0)}, {F1_keypoint}\n\n")
        # print(np.array(topo_results))
        with open(txt_path_checkpoint, mode="a") as f:
            f.write(f"all num: {all_num}, tree num: {tree_num}\n\n")

if __name__ == '__main__':
    import argparse

    is_test_use_mst = True
    # is_test_use_mst = False


    def get_args_parser(is_test_use_mst=False):
        parser = argparse.ArgumentParser('Set testing param', add_help=False)
        ####################
        check_folder = r"D:\pythonProject\lab_training\6_bbox\relationformer\road_final_12_hard_2023_syn\new_infinity_500_smd\guyot_squid\20240729_gradmst\villier_softmax_gradmst\20240717_guyot_subset_epsilon_change_neg_infinity_500smd_gradmst\trained_weights"
        check_server = "runs/3407"
        check_num = 203  # 43

        print(f"check_server: {check_server}, check_num: {check_num}")

        if is_test_use_mst:
            # check_test_mst = 'test_mst_dist_networkx'
            check_test_mst = 'valid_mst2000'
        else:
            check_test_mst = 'valid_unmst2000'
        parser.add_argument('--config',
                            # default=f'{src_folder}/trained_weights/{check_folder}/configs/tree_2D_use_mst.yaml',
                            default=fr'D:\pythonProject\lab_training\6_bbox\relationformer\road_final_12_hard_2023_syn\new_infinity_500_smd\guyot_squid\20240729_gradmst\villier_softmax_gradmst\20240717_guyot_subset_epsilon_change_neg_infinity_500smd_gradmst\configs\tree_2D_use_mst_only1.yaml',
                            help='config file (.yml) containing the hyper-parameters for training. '
                                 'If None, use the nnU-Net config. See /config for examples.')
        parser.add_argument('--checkpoint',
                        # default=f'{src_folder}/trained_weights/{check_folder}/{check_server}/checkpoint_{check_num}_epoch.pkl',
                        default=fr"D:\pythonProject\lab_training\6_bbox\relationformer\road_final_12_hard_2023_syn\new_infinity_500_smd\guyot_squid\20240729_gradmst\villier_softmax_gradmst\20240717_guyot_subset_epsilon_change_neg_infinity_500smd_gradmst\trained_weights\runs\3407\checkpoint_203_epoch.pkl",
                        help='checkpoint of the model to test.')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training')
        parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0, 1],
                            help='list of index where skip conn will be made')
        parser.add_argument('--visualize',
                            # default=f'{src_folder}/trained_weights/{check_folder}/{check_server}/{check_num}/{check_test_mst}',
                            default=fr'{check_folder}/{check_server}/{check_num}/{check_test_mst}',
        #                   default=None,
                            help='path to save visualized graphs to.')
        parser.add_argument('--use_gnn', default=False, help='use gnn')
        parser.add_argument('--use_mst_train', default=True, help='use mst train')
        return parser
    parser = argparse.ArgumentParser('test Relationformer', parents=[get_args_parser(is_test_use_mst=is_test_use_mst)])
    args = parser.parse_args()
    test(is_use_mst=is_test_use_mst, args=args)

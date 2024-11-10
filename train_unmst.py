import os
import random
import yaml
# import sys
import json
# from argparse import ArgumentParser
import numpy as np

import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, sampler
from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as TF

########################################################################################################################
from skimage import exposure
import networkx as nx
import cv2

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
        self.file_list = self.processed_file_names

        ids1, (list_DETR_points_left_up, list_DETR_node_collections) = load_detr_dataset(self.tgt_data_path)
        self.ids1 = ids1
        self.list_DETR_points_left_up = list_DETR_points_left_up
        self.list_DETR_node_collections = list_DETR_node_collections

        self.max_size = max_size

        self.max_change_light_rate = max_change_light_rate
        self.is_train = is_train
        self.is_rotate = is_rotate


    @property
    def processed_file_names(self):
        path_list = []
        for file in os.listdir(self.tgt_data_path):
            if file.endswith(".pt"):
                path_list.append(file)
        return path_list
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
        final_rotate_nodes_tensor = torch.tensor(rotate_nodes_data, dtype=torch.float32, device=nodes_tensor.device) / torch.tensor(
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

        for x, y in (rotate_nodes_tensor * torch.tensor([width, height], device=rotate_nodes_tensor.device)).cpu().numpy():
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
        list_DETR_points_left_up_idx = self.list_DETR_points_left_up[idx]
        list_DETR_node_collections_idx = self.list_DETR_node_collections[idx]

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
        else:
            feature_img = input_img
            nodes = list_DETR_points_left_up_idx
        list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)

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
                # 按需生成 PAFs, mask, unet, heatmap
                feature_size = (feature_img.shape[1], feature_img.shape[2])
                PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
                    list_DETR_node_collections_idx=list_DETR_node_collections_idx,
                    list_DETR_points_left_up_idx=list_DETR_points_left_up,
                    feature_size=feature_size,
                    sigma=3, unet_thickness=3, mask_thickness=6
                )
            else:
                # 按需生成 PAFs, mask, unet, heatmap
                # 并且要按照旋转后的坐标系生成
                feature_size = (feature_img.shape[1], feature_img.shape[2])
                PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
                    list_DETR_node_collections_idx=list_DETR_node_collections_idx,
                    list_DETR_points_left_up_idx=list_DETR_points_left_up,
                    feature_size=feature_size,
                    sigma=3, unet_thickness=3, mask_thickness=6
                )
                dsize = (feature_size[1], feature_size[0])
                PAFs_idx = self._rotate_tensor(PAFs_idx, M, dsize)
                mask_idx = self._rotate_tensor(mask_idx, M, dsize)
                unet_idx = self._rotate_tensor(unet_idx, M, dsize)
                heatmap_idx = self._rotate_tensor(heatmap_idx, M, dsize)
        else:
            # 按需生成 PAFs, mask, unet, heatmap
            feature_size = (feature_img.shape[1], feature_img.shape[2])
            PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
                list_DETR_node_collections_idx=list_DETR_node_collections_idx,
                list_DETR_points_left_up_idx=list_DETR_points_left_up,
                feature_size=feature_size,
                sigma=3, unet_thickness=3, mask_thickness=6
            )


        return (feature_img.contiguous(), label_img_name0,
                list_DETR_points_left_up, list_DETR_node_collections_idx,
                PAFs_idx, mask_idx, unet_idx, heatmap_idx,
                self.ids1[idx])

########################################################################################################################
def custom_collate_fn(batch):
    (feature_img, label_img_name0, list_DETR_points_left_up, list_DETR_node_collections,
     list_PAFs, list_mask, list_unet, list_heatmap,
     ids1) = zip(*batch)
    ACT_1 = 0.9999999
    ACT_0 = 0.0000001

    # images = torch.cat([item for item in feature_img], 0).contiguous()
    # 这里改为 是一个列表  每个图像被分别存进去
    images = [item.to(torch.float32) for item in feature_img]

    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]

    PAFs_list_transformed = [PAFs.unsqueeze(0).permute(0, 3, 1, 2) for PAFs in list_PAFs]
    mask_list_transformed = [mask.unsqueeze(0).unsqueeze(0) for mask in list_mask]
    unet_list_transformed = [unet.unsqueeze(0).unsqueeze(0) for unet in list_unet]
    heatmap_list_transformed = [heatmap.unsqueeze(0).unsqueeze(0) for heatmap in list_heatmap]

    PAFs_concatenated = torch.cat(PAFs_list_transformed, 0)  # 尺寸变为[batch_size, 2, 570, 190]
    # mask_concatenated = torch.cat(mask_list_transformed, 0).to(torch.float32)  # 尺寸变为[batch_size, 1, 570, 190]
    mask_concatenated = torch.cat(mask_list_transformed, 0).contiguous()  # 尺寸变为[batch_size, 1, 570, 190]
    unet_concatenated = torch.cat(unet_list_transformed, 0)  # 尺寸变为[batch_size, 1, 570, 190]
    heatmap_concatenated = torch.cat(heatmap_list_transformed, 0)  # 尺寸变为[batch_size, 1, 570, 190]

    PAFs_concatenated = torch.clamp(PAFs_concatenated, min=-ACT_1, max=ACT_1).contiguous()  # 范围限制在[-1, 1]
    # mask_concatenated = torch.clamp(mask_concatenated, min=ACT_0, max=ACT_1).contiguous()  # 范围限制在[0, 0.99999]
    unet_concatenated = torch.clamp(unet_concatenated, min=ACT_0, max=ACT_1).contiguous()  # 范围限制在[0, 0.99999]
    heatmap_concatenated = torch.clamp(heatmap_concatenated, min=ACT_0, max=ACT_1).contiguous()  # 范围限制在[0, 0.99999]

    detr_ids = list(ids1)
    return [images, points_left_up, edges,
            PAFs_concatenated, mask_concatenated, unet_concatenated, heatmap_concatenated,
            detr_ids],

########################################################################################################################
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    # Load the config files
    # import torch
    import torch.distributed as dist

    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size():
        if not is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank():
        if not is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def get_local_size():
        if not is_dist_avail_and_initialized():
            return 1
        return int(os.environ['LOCAL_SIZE'])

    def get_local_rank():
        if not is_dist_avail_and_initialized():
            return 0
        return int(os.environ['LOCAL_RANK'])


    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    local_rank = args.local_rank
    # dist.init_process_group(backend='gloo') # windows
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", local_rank)

    # import logging
    # from monai.data import DataLoader
    # from monai.engines import SupervisedTrainer
    # from monai.handlers import MeanDice, StatsHandler
    # from monai.inferers import SimpleInferer
    # from dataset_road_network import build_road_network_data

    # fix the seed for reproducibility
    seed = config.DATA.SEED + get_rank()
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.use_gnn:
        print("use gnn")
        # from evaluator_gnn import build_evaluator
    elif args.use_mst_train:
        print("use mst edge")
        # from evaluator import build_evaluator
    else:
        print("use edge")
        # from evaluator import build_evaluator


    # from trainer import build_trainer
    from models import build_model
    # from monai.losses import DiceCELoss
    # from utils import image_graph_collate_road_network

    # from tensorboardX import SummaryWriter
    from models.matcher import build_matcher
    from losses_only import SetCriterion
    from epoch import epoch_train, epoch_val
    import time
    from metric_smd import StreetMoverDistance
    from monai.utils import MetricReduction

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    # img_train_dataset_path1 = "/mnt/datasets/root_dataset/final/train50/img"
    # img_train_dataset_path2 = "/mnt/datasets/root_dataset/final/train50/unet"
    # tgt_train_dataset_path = "/mnt/datasets/root_dataset/final/train50/data"

    # img_train_dataset_path1 = "/mnt/workspace2022/liu/datasets/100_30_random_3/train/branch"
    # img_train_dataset_path2 = "/mnt/workspace2022/liu/datasets/100_30_random_3/train/branch_node"
    # tgt_train_dataset_path = "/mnt/workspace2022/liu/datasets/100_30_random_3/train/data"

    # img_train_dataset_path1 = "/mnt/workspace2022/liu/datasets/new_8/train/img"
    # img_train_dataset_path2 = "/mnt/workspace2022/liu/datasets/new_8/train/unet"
    # tgt_train_dataset_path = "/mnt/workspace2022/liu/datasets/new_8/train/data"

    # img_train_dataset_path1 = "/mnt/datasets/root_dataset/individuals_net/train/img"
    # img_train_dataset_path2 = "/mnt/datasets/root_dataset/individuals_net/train/unet"
    # tgt_train_dataset_path = "/mnt/datasets/root_dataset/individuals_net/train/data"

    # img_test_dataset_path1 = "/mnt/datasets/root_dataset/individuals_max_12/test/img"
    # img_test_dataset_path2 = "/mnt/datasets/root_dataset/individuals_max_12/test/unet" individuals_net
    # tgt_test_dataset_path = "/mnt/datasets/root_dataset/individuals_max_12/test/data"

    # img_val_dataset_path1 = "/mnt/datasets/root_dataset/final/val/img"
    # img_val_dataset_path2 = "/mnt/datasets/root_dataset/final/val/unet"
    # tgt_val_dataset_path = "/mnt/datasets/root_dataset/final/val/data"

    # img_val_dataset_path1 = "/mnt/datasets/root_dataset/individuals_net/val/img"
    # img_val_dataset_path2 = "/mnt/datasets/root_dataset/individuals_net/val/unet"
    # tgt_val_dataset_path = "/mnt/datasets/root_dataset/individuals_net/val/data"

    # img_val_dataset_path1 = "/mnt/workspace2022/liu/datasets/100_30_random_3/valid/branch"
    # img_val_dataset_path2 = "/mnt/workspace2022/liu/datasets/100_30_random_3/valid/branch_node"
    # tgt_val_dataset_path = "/mnt/workspace2022/liu/datasets/100_30_random_3/valid/data"

    # img_val_dataset_path1 = "/mnt/workspace2022/liu/datasets/new_8/val_aug/img"
    # img_val_dataset_path2 = "/mnt/workspace2022/liu/datasets/new_8/val_aug/unet"
    # tgt_val_dataset_path = "/mnt/workspace2022/liu/datasets/new_8/val_aug/data"

    # train_path = "/mnt/workspace2022/liu/datasets/new_8/train"
    # train_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/new_8_new/train"
    # train_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/move_data/train_aug"

    # val_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/new_8_new/val_aug"
    # val_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/move_data/val_aug"

    train_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/all_same_PAF_move/train_aug"
    val_path = "/sqfs2/cmc/1/work/G15538/u6c043/data/dataset/all_same_PAF_move/val_aug"

    # dataset_train = LoadCNNDataset(tgt_data_path=tgt_train_dataset_path, feature_path_1=img_train_dataset_path1,
    #                                feature_path_2=img_train_dataset_path2, tgt_detr_dataset_name="DETR_all_",
    #                                tgt_gnn_dataset_name="GNN_simple_")
    # dataset_test = LoadCNNDataset(tgt_data_path=tgt_test_dataset_path, feature_path_1=img_test_dataset_path1,
    #                               feature_path_2=img_test_dataset_path2)
    # dataset_val = LoadCNNDataset(tgt_data_path=tgt_val_dataset_path, feature_path_1=img_val_dataset_path1,
    #                              feature_path_2=img_val_dataset_path2, tgt_detr_dataset_name="DETR_all_",
    #                              tgt_gnn_dataset_name="GNN_simple_")

    dataset_train = LoadCNNDataset(parent_path=train_path, max_size=config.DATA.MAX_SIZE, max_change_light_rate=0.3, is_train=False, is_rotate=True)
    dataset_val = LoadCNNDataset(parent_path=val_path, max_size=config.DATA.MAX_SIZE, max_change_light_rate=0.3, is_train=False, is_rotate=False)

    # train_indices = list(range(len(dataset_train)))
    # random.shuffle(train_indices)
    # dataset_train = torch.utils.data.Subset(dataset_train, train_indices[:160])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    train_loader = DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE, shuffle=False,  ######
                              collate_fn=custom_collate_fn, drop_last=True, pin_memory=True,
                              num_workers=4,
                              sampler=train_sampler)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
    #                              collate_fn=custom_collate_fn, drop_last=True, pin_memory=True,
    #                                   num_workers=config.DATA.NUM_WORKERS)

    # val_indices = range(len(dataset_val))
    # dataset_val = torch.utils.data.Subset(dataset_val, val_indices[:1000])
    valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)

    val_loader = DataLoader(dataset_val, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
                            collate_fn=custom_collate_fn, drop_last=True, pin_memory=True,
                            num_workers=4,
                            sampler=valid_sampler)
    if dist.get_rank() == 0:
        print("Dataset splits -> Train: {} | Valid: {}\n".format(len(dataset_train), len(dataset_val)))

    net = build_model(config, args).to(device)

    param_dicts = [
        {
            "params":
                [p for n, p in net.named_parameters()
                 if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points',
                                                                                              'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE)
        },
        {
            "params": [p for n, p in net.named_parameters() if
                       match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR) * 0.1
        }
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)
    last_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["net"].items():
            name = k[7:]  # remove `module.`
            # print(k, name)
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)

        # net.load_state_dict(checkpoint["net2"])

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = scheduler.last_epoch
        print(last_epoch)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device=device)
    # net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True,
    #                                                 device_ids=[local_rank], output_device=local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    matcher = build_matcher(config)
    loss = SetCriterion(config=config, matcher=matcher, net=net, args=args)
    val_smd_loss = []
    train_total_loss, train_class_loss, train_nodes_loss, train_edges_loss, train_boxes_loss, train_cards_loss = [], [], [], [], [], []
    SMD = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    epochs = config.TRAIN.EPOCHS - last_epoch + 1
    check_path = config.TRAIN.SAVE_PATH + "/runs/" + config.log.exp_name + "_" + str(config.DATA.SEED) + "/"
    if dist.get_rank() == 0:
        os.makedirs(check_path, exist_ok=True)
        print(check_path)

    # ===================random guessing====================
    # if not args.resume:
    #     first_start = time.time()
    #     smd_loss = epoch_val(val_loader=val_loader, net=net, config=config, device=device, SMD=SMD, args=args)
    #     first_end = time.time() - first_start
    #     if dist.get_rank() == 0:
    #         print(
    #             'Epoch {}/{} || Train total: {:.4f} class: {:.4f} nodes: {:.4f} edges: {:.4f} boxes: {:.4f} cards: {:.4f} || Val smd: {:.4f} '
    #             ' take {:.4f} sec.'
    #                 .format(0, epochs, 0, 0, 0, 0, 0, 0, smd_loss, first_end))

    # ========================train=========================
    min_loss_valid = (10000000, 0)
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_total, train_class, train_nodes, train_edges, train_boxes, train_cards = \
            epoch_train(train_loader=train_loader, net=net, loss_function=loss, optimizer=optimizer,
                        device=device, last_epoch=last_epoch, epoch_now=last_epoch + 1 + epoch, max_epoch=config.TRAIN.EPOCHS)
        smd_loss = epoch_val(val_loader=val_loader, net=net, config=config, device=device, SMD=SMD, args=args)
        # ========================log and plot=========================
        epoch_time = time.time() - epoch_start
        if dist.get_rank() == 0:
            print(
                'Epoch {}/{} || Train total: {:.4f} class: {:.4f} nodes: {:.4f} edges: {:.4f} boxes: {:.4f} cards: {:.4f} || Val smd: {:.8f} '
                ' take {:.4f} sec.'
                    .format(epoch + 1, epochs, train_total, train_class, train_nodes, train_edges, train_boxes, train_cards,
                            smd_loss, epoch_time))
        val_smd_loss.append(smd_loss)
        train_total_loss.append(train_total)
        train_class_loss.append(train_class)
        train_nodes_loss.append(train_nodes)
        train_edges_loss.append(train_edges)
        train_boxes_loss.append(train_boxes)
        train_cards_loss.append(train_cards)
        # =========================save models=========================
        if min_loss_valid[0] > smd_loss and dist.get_rank() == 0:
            min_loss_valid = (smd_loss, epoch + 1)
            checkpoint = {"net": net.state_dict(),  # 模型数据
                          "net2": net.module.state_dict(),  # 模型数据
                          "optimizer": optimizer.state_dict(),  # 优化器数据
                          'scheduler': scheduler.state_dict(),
                          }
            path_checkpoint = check_path + 'checkpoint_{}_epoch.pkl'.format(epoch + 1 + last_epoch)
            torch.save(checkpoint, path_checkpoint)
            np_path_checkpoint = check_path + 'checkpoint_{}_epoch.npz'.format(epoch + 1 + last_epoch)
            np.savez(np_path_checkpoint,
                     val_smd_loss=np.array(val_smd_loss),
                     train_total_loss=np.array(train_total_loss),
                     train_class_loss=np.array(train_class_loss),
                     train_nodes_loss=np.array(train_nodes_loss),
                     train_edges_loss=np.array(train_edges_loss),
                     train_boxes_loss=np.array(train_boxes_loss),
                     train_cards_loss=np.array(train_cards_loss)
                     )
            txt_path_checkpoint = check_path + '{}_epoch_{}_smd.txt'.format(epoch + 1 + last_epoch, smd_loss)
            open(txt_path_checkpoint, "a")
            print("save models: {}".format(epoch + 1 + last_epoch))

        elif (epoch + 1 + last_epoch) % 10 == 0 and dist.get_rank() == 0:
            checkpoint = {"net": net.state_dict(),  # 模型数据
                          "net2": net.module.state_dict(),  # 模型数据
                          "optimizer": optimizer.state_dict(),  # 优化器数据
                          'scheduler': scheduler.state_dict(),
                          }
            path_checkpoint = check_path + 'checkpoint_{}_epoch.pkl'.format(epoch + 1 + last_epoch)
            torch.save(checkpoint, path_checkpoint)
            np_path_checkpoint = check_path + 'checkpoint_{}_epoch.npz'.format(epoch + 1 + last_epoch)
            np.savez(np_path_checkpoint,
                     val_smd_loss=np.array(val_smd_loss),
                     train_total_loss=np.array(train_total_loss),
                     train_class_loss=np.array(train_class_loss),
                     train_nodes_loss=np.array(train_nodes_loss),
                     train_edges_loss=np.array(train_edges_loss),
                     train_boxes_loss=np.array(train_boxes_loss),
                     train_cards_loss=np.array(train_cards_loss)
                     )
            txt_path_checkpoint = check_path + '{}_epoch_{}_smd.txt'.format(epoch + 1 + last_epoch, smd_loss)
            open(txt_path_checkpoint, "a")
            print("save models: {}".format(epoch + 1 + last_epoch))

        scheduler.step()
    if dist.get_rank() == 0:
        checkpoint = {"net": net.state_dict(),  # 模型数据
                      "net2": net.module.state_dict(),  # 模型数据
                      "optimizer": optimizer.state_dict(),  # 优化器数据
                      'scheduler': scheduler.state_dict(),
                      }
        path_checkpoint = check_path + 'checkpoint_{}_epoch.pkl'.format(epochs)
        torch.save(checkpoint, path_checkpoint)
        np_path_checkpoint = check_path + 'checkpoint_{}_epoch.npz'.format(epochs)
        np.savez(np_path_checkpoint,
                 val_smd_loss=np.array(val_smd_loss),
                 train_total_loss=np.array(train_total_loss),
                 train_class_loss=np.array(train_class_loss),
                 train_nodes_loss=np.array(train_nodes_loss),
                 train_edges_loss=np.array(train_edges_loss),
                 train_boxes_loss=np.array(train_boxes_loss),
                 train_cards_loss=np.array(train_cards_loss)
                 )
    print("\nTraining Completed!")
    print("Minimum loss on validation set: {} at epoch {}".format(min_loss_valid[0], min_loss_valid[1]))

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_args_parser():
    parser = argparse.ArgumentParser('Set training param', add_help=False)
    parser.add_argument('--config', default='configs/tree_2D_use_mst.yaml',
                        help='config file (.yml) containing the hyper-parameters for training. '
                             'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')
    parser.add_argument('--use_gnn', default=False, help='use gnn')
    parser.add_argument('--use_mst_train', default=False, help='use mst train')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training Relationformer', parents=[get_args_parser()])
    args = parser.parse_args()

    # import torch
    # torch.autograd.set_detect_anomaly(True)

    # import torch.multiprocessing
    #
    # torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)

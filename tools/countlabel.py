from pycocotools.coco import COCO
import numpy as np

# 替换为你的COCO格式注释文件的路径
coco_annotation_file = '/home/duomeitinrfx/data/coco/annotations/instances_train2017.json'

# 加载COCO数据集
coco = COCO(coco_annotation_file)

# 获取所有类别的ID
category_ids = coco.getCatIds()

# 计算每个类别的标签数量
category_label_counts = [len(coco.getAnnIds(catIds=[cat_id])) for cat_id in category_ids]

# 计算平均、最大和最小标签数量
average_labels = np.mean(category_label_counts)
max_labels = np.max(category_label_counts)
min_labels = np.min(category_label_counts)

print(f"平均每个类别的标签数量: {average_labels}")
print(f"最多标签数量的类别: {max_labels}")
print(f"最少标签数量的类别: {min_labels}")


from xml.etree import ElementTree as ET
import os
import numpy as np
from collections import defaultdict

# 假定一个函数用于解析单个Pascal VOC的XML文件
# def parse_voc_xml(file_path):
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     objects = root.findall('object')
#     labels = [obj.find('name').text for obj in objects]
#     return labels
#
# # 假定的Pascal VOC标注文件夹路径
# annotations_dir = '/home/duomeitinrfx/data/VOC/VOCdevkit/VOC2012/Annotations'
#
# # 读取目录下的所有XML文件
# annotation_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.xml')]
#
# # 统计每个类别的标签数量
# category_counts = defaultdict(int)
#
# for file_path in annotation_files:
#     labels = parse_voc_xml(file_path)
#     for label in labels:
#         category_counts[label] += 1
#
# # 计算平均、最大、最小标签数量
# label_counts = np.array(list(category_counts.values()))
# average_count = np.mean(label_counts)
# max_count = np.max(label_counts)
# min_count = np.min(label_counts)
#
# print(f"平均每个类别的标签数量: {average_count}")
# print(f"最多标签数量的类别: {max_count}")
# print(f"最少标签数量的类别: {min_count}")
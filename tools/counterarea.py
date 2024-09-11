# from pycocotools.coco import COCO
# import os
#
# # COCO数据集的JSON文件路径
# coco_json_file = '/home/duomeitinrfx/data/coco/annotations/instances_train2017.json'
#
#
# # 初始化COCO api
# coco = COCO(coco_json_file)
#
# # 用于累计所有目标的面积比例
# total_area_ratio = 0.0
#
# # 初始化最大和最小面积比例的变量
# max_area_ratio = 0.0
# min_area_ratio = float('inf')  # 设置为无穷大，确保任何实际比例都会更小
#
# # 计数器，用于记录处理的目标数量
# object_count = 0
#
# # 遍历所有图像
# for img_id in coco.imgs:
#     img_info = coco.imgs[img_id]
#     img_area = img_info['width'] * img_info['height']
#
#     # 获取当前图像的所有标注
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)
#
#     # 遍历每个标注，计算面积比例
#     for ann in anns:
#         bbox = ann['bbox']  # bbox格式：[x, y, width, height]
#         bbox_area = bbox[2] * bbox[3]
#         area_ratio = bbox_area / img_area
#
#         # 更新总面积比例
#         total_area_ratio += area_ratio
#
#         # 更新最大和最小面积比例
#         if area_ratio > max_area_ratio and area_ratio!=1:
#             max_area_ratio = area_ratio
#         if area_ratio < min_area_ratio and area_ratio!=0:
#             min_area_ratio = area_ratio
#
#         object_count += 1
#
# # 计算平均面积比例
# average_area_ratio = total_area_ratio / object_count if object_count else 0
#
# print(f"Average object area ratio: {average_area_ratio}")
# print(f"Maximum object area ratio: {max_area_ratio}")
# print(f"Minimum object area ratio: {min_area_ratio}")
#

import os
import xml.etree.ElementTree as ET

# # Pascal VOC数据集的标注文件所在目录
# annotations_dir = '/home/duomeitinrfx/data/VOC/VOCdevkit/VOC2007/Annotations'
#
# # 初始化统计变量
# total_area_ratio = 0.0
# max_area_ratio = 0.0
# min_area_ratio = float('inf')
# object_count = 0
#
# # 遍历标注目录中的所有XML文件
# for filename in os.listdir(annotations_dir):
#     if not filename.endswith('.xml'):
#         continue
#
#     xml_path = os.path.join(annotations_dir, filename)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     # 获取图像的宽度和高度
#     size = root.find('size')
#     img_width = int(size.find('width').text)
#     img_height = int(size.find('height').text)
#     img_area = img_width * img_height
#
#     # 遍历所有目标对象
#     for obj in root.iter('object'):
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)
#
#         # 计算目标的面积
#         object_area = (xmax - xmin) * (ymax - ymin)
#         area_ratio = object_area / img_area
#
#         # 更新统计数据
#         total_area_ratio += area_ratio
#         object_count += 1
#         if area_ratio > max_area_ratio:
#             max_area_ratio = area_ratio
#         if area_ratio < min_area_ratio:
#             min_area_ratio = area_ratio
#
# # 计算平均面积比例
# average_area_ratio = total_area_ratio / object_count if object_count else 0
#
# print(f"Average object area ratio: {average_area_ratio:.4f}")
# print(f"Maximum object area ratio: {max_area_ratio:.4f}")
# print(f"Minimum object area ratio: {min_area_ratio:.4f}")

import xml.etree.ElementTree as ET
import os

# 假设数据集的路径
annotations_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations'


# 用于存储结果的字典
class_areas = {}  # 每个类别的总面积
class_ratios = {}  # 每个类别的总占比
class_counts = {}  # 每个类别的对象计数

total_area = 0  # 所有对象的总面积
total_ratio = 0  # 所有对象的总占比
total_count = 0  # 总对象数

# 遍历标注文件
for filename in os.listdir(annotations_path):
    if not filename.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(annotations_path, filename))
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image_area = width * height

    for obj in root.findall('object'):
        cls = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        area = (xmax - xmin) * (ymax - ymin)
        ratio = area / image_area

        class_areas[cls] = class_areas.get(cls, 0) + area
        class_ratios[cls] = class_ratios.get(cls, 0) + ratio
        class_counts[cls] = class_counts.get(cls, 1) + 1

        total_area += area
        total_ratio += ratio
        total_count += 1

# 计算平均值
avg_class_areas = {cls: area / class_counts[cls] for cls, area in class_areas.items()}
avg_class_ratios = {cls: ratio / class_counts[cls] for cls, ratio in class_ratios.items()}
avg_total_area = total_area / total_count
avg_total_ratio = total_ratio / total_count

print("Average pixel area per class:", avg_class_areas)
print("Average ratio per class:", avg_class_ratios)
print("Average pixel area for all objects:", avg_total_area)
print("Average ratio for all objects:", avg_total_ratio)





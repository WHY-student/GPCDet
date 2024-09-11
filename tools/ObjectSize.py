# import xml.etree.ElementTree as ET
# import os
#
#
# # 定义函数来解析Pascal VOC格式的XML注释文件
# def parse_annotation(xml_file_path):
#     tree = ET.parse(xml_file_path)
#     root = tree.getroot()
#
#     # 获取图像的宽度和高度
#     size_elem = root.find("size")
#     img_width = int(size_elem.find("width").text)
#     img_height = int(size_elem.find("height").text)
#
#     # 统计小目标和大目标的数量
#     small_target_count = 0
#     large_target_count = 0
#
#     for obj_elem in root.findall("object"):
#         bbox_elem = obj_elem.find("bndbox")
#         xmin = int(bbox_elem.find("xmin").text)
#         ymin = int(bbox_elem.find("ymin").text)
#         xmax = int(bbox_elem.find("xmax").text)
#         ymax = int(bbox_elem.find("ymax").text)
#
#         # 计算目标的宽度和高度
#         obj_width = xmax - xmin
#         obj_height = ymax - ymin
#
#         # 计算目标占据图片尺寸的百分比
#         obj_area = obj_width * obj_height
#         img_area = img_width * img_height
#         obj_percentage = (obj_area / img_area) * 100
#
#         # 根据定义判断目标是小目标还是大目标
#         if obj_percentage <= 0.58:
#             small_target_count += 1
#         else:
#             large_target_count += 1
#
#     return small_target_count, large_target_count
#
#
# # 定义函数来统计整个数据集中小目标和大目标的数量
# def count_small_large_targets(dataset_dir):
#     small_target_total = 0
#     large_target_total = 0
#
#     # 遍历数据集目录下的所有XML注释文件
#     for filename in os.listdir(dataset_dir):
#         if filename.endswith(".xml"):
#             xml_file_path = os.path.join(dataset_dir, filename)
#             small_target_count, large_target_count = parse_annotation(xml_file_path)
#             small_target_total += small_target_count
#             large_target_total += large_target_count
#
#     return small_target_total, large_target_total
#
#
# 指定Pascal VOC数据集的路径
# dataset_dir = "/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations"
#
# # 统计小目标和大目标的数量
# small_target_total, large_target_total = count_small_large_targets(dataset_dir)
#
# # 打印结果
# print("小目标数量:", small_target_total)
# print("大目标数量:", large_target_total)
#
# print(small_target_total/(small_target_total+large_target_total))

import os
import xml.etree.ElementTree as ET

#定义函数来解析Pascal VOC格式的XML注释文件并判断目标是否为小目标
def is_small_target(xml_file_path, threshold=0.1):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 获取图像的宽度和高度
    size_elem = root.find("size")
    img_width = int(size_elem.find("width").text)
    img_height = int(size_elem.find("height").text)

    # 遍历所有目标
    for obj_elem in root.findall("object"):
        bbox_elem = obj_elem.find("bndbox")
        xmin = int(bbox_elem.find("xmin").text)
        ymin = int(bbox_elem.find("ymin").text)
        xmax = int(bbox_elem.find("xmax").text)
        ymax = int(bbox_elem.find("ymax").text)

        # 计算目标的宽度和高度
        target_width = xmax - xmin
        target_height = ymax - ymin

        # 计算目标的相对尺寸
        relative_width = target_width / img_width
        relative_height = target_height / img_height

        # 判断目标是否为小目标
        if relative_width <= threshold and relative_height <= threshold:
            return True

    return False

# 指定Pascal VOC数据集的路径
dataset_dir = "/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations"

# 统计小目标和大目标的数量
small_target_count = 0
large_target_count = 0

# 遍历数据集目录下的所有XML注释文件
for filename in os.listdir(dataset_dir):
    if filename.endswith(".xml"):
        xml_file_path = os.path.join(dataset_dir, filename)
        if is_small_target(xml_file_path, threshold=0.10):
            small_target_count += 1
        else:
            large_target_count += 1

# 计算占比
total_target_count = small_target_count + large_target_count
small_target_percentage = (small_target_count / total_target_count) * 100
large_target_percentage = (large_target_count / total_target_count) * 100

# 打印结果
print("小目标占比: {:.2f}%".format(small_target_percentage))
print("大目标占比: {:.2f}%".format(large_target_percentage))

import os
import xml.etree.ElementTree as ET

# 定义目标大小的阈值
small_threshold = 32 * 32
medium_threshold = 96 * 96
#
# # 数据集路径
dataset_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007'

# 初始化大小统计变量
small_count = 0
medium_count = 0
large_count = 0

# 遍历数据集中的每个标注文件
for filename in os.listdir(os.path.join(dataset_path, 'Annotations')):
    if filename.endswith('.xml'):
        xml_path = os.path.join(dataset_path, 'Annotations', filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 解析XML文件以获取目标的宽度和高度
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            width = int(bbox.find('xmax').text) - int(bbox.find('xmin').text)
            height = int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
            area = width * height

            # 根据目标大小分别增加统计变量
            if area <= small_threshold:
                small_count += 1
            elif small_threshold < area <= medium_threshold:
                medium_count += 1
            else:
                large_count += 1

# 打印统计结果
total_count = small_count + medium_count + large_count
print(f"小目标数量：{small_count}，占比：{small_count / total_count:.2%}")
print(f"中等目标数量：{medium_count}，占比：{medium_count / total_count:.2%}")
print(f"大目标数量：{large_count}，占比：{large_count / total_count:.2%}")

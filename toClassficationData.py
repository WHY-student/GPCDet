# import os
# import cv2
# import xml.etree.ElementTree as ET
# from shutil import copyfile
#
# # 输入路径
# voc_data_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations'
# img_path='/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages'
# output_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/classfication'
#
# # 创建输出文件夹
# os.makedirs(output_path, exist_ok=True)
#
# # 解析Pascal VOC标注文件
# def parse_annotation(xml_file):
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     filename = root.find('filename').text
#     objects = root.findall('object')
#
#     annotations = []
#     for obj in objects:
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)
#         label = obj.find('name').text
#
#         annotations.append({'filename': filename, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'label': label})
#
#     return annotations
#
# # 切割目标实例并保存
# def crop_and_save(image_path, annotation, output_folder, instance_count):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to load image: {image_path}")
#         return  # 或者采取其他处理方式
#     xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
#     cropped_img = img[ymin:ymax, xmin:xmax]
#
#     if cropped_img is None:
#         return
#     # 为图片添加编号并保存
#     output_filename = os.path.join(output_folder, f"{annotation['filename'].split('.')[0]}_{annotation['label']}_{instance_count}.jpg")
#     print(output_filename)
#     cv2.imwrite(output_filename, cropped_img)
#
# # 读取每个图像的标注文件并执行切割
# k=0
# for xml_file in os.listdir(voc_data_path):
#     if xml_file.endswith(".xml"):
#         k+=1
#         print(k)
#         xml_path = os.path.join(voc_data_path, xml_file)
#         annotations = parse_annotation(xml_path)
#
#         for i, annotation in enumerate(annotations, start=1):
#             image_path = os.path.join(img_path, annotation['filename'])
#             crop_and_save(image_path, annotation, output_path, i)


import os
import cv2
import xml.etree.ElementTree as ET
from shutil import copyfile
from tqdm import tqdm

# 输入路径
voc_data_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007'
output_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/classfication'

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

# 解析Pascal VOC标注文件
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    objects = root.findall('object')

    annotations = []
    for obj in objects:
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        label = obj.find('name').text

        annotations.append({'filename': filename, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'label': label})

    return annotations

# 切割目标实例并保存
def crop_and_save(image_path, annotation, output_folder, instance_count):
    img = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
    cropped_img = img[ymin:ymax, xmin:xmax]

    # 为图片添加编号并保存
    output_filename = os.path.join(output_folder, f"{annotation['filename'].split('.')[0]}_{annotation['label']}_{instance_count}.jpg")
    cv2.imwrite(output_filename, cropped_img)

# 读取每个图像的标注文件并执行切割
image_sets_path = os.path.join(voc_data_path, '0.8_0.2_Sets', 'Main')
train_file = os.path.join(image_sets_path, 'train.txt')
train_images = [line.strip() for line in open(train_file).readlines()]


val_file = os.path.join(image_sets_path, 'val.txt')
val_images = [line.strip() for line in open(val_file).readlines()]
# 读取每个图像的标注文件并执行切割
for xml_file in tqdm(os.listdir(os.path.join(voc_data_path, 'Annotations')), desc='Processing Annotations'):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(voc_data_path, 'Annotations', xml_file)
        annotations = parse_annotation(xml_path)

        for i, annotation in enumerate(annotations, start=1):
            image_filename = annotation['filename'].split('.')[0]  # 去掉文件扩展名
            image_path = os.path.join(voc_data_path, 'JPEGImages', annotation['filename'])
            output_folder = os.path.join(output_path, 'train' if image_filename in train_images else 'val', annotation['label'])
            os.makedirs(output_folder, exist_ok=True)
            crop_and_save(image_path, annotation, output_folder, i)



import os
import xml.etree.ElementTree as ET
import cv2

# def draw_bounding_boxes_and_labels(data_dir, output_folder ):
#     annotations_folder = os.path.join(data_dir, 'Annotations')
#     image_folder = os.path.join(data_dir, 'JPEGImages')
#     # 遍历注释文件夹中的所有 XML 文件
#     for xml_file in os.listdir(annotations_folder):
#         if xml_file.endswith('.xml'):
#             xml_path = os.path.join(annotations_folder, xml_file)
#             image_path = os.path.join(image_folder, xml_file.replace('.xml', '.jpg'))
#             image = cv2.imread(image_path)
#
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#
#             for obj in root.findall('object'):
#                 # 获取类别名称
#                 category = obj.find('name').text
#
#                 # 获取真实框坐标
#                 bndbox = obj.find('bndbox')
#                 xmin = int(bndbox.find('xmin').text)
#                 ymin = int(bndbox.find('ymin').text)
#                 xmax = int(bndbox.find('xmax').text)
#                 ymax = int(bndbox.find('ymax').text)
#
#                 # 绘制矩形（真实框）
#                 cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
#                 #文字背景
#                 (text_width, text_height), baseline = cv2.getTextSize(category, cv2.FONT_ITALIC, 1.6, 2)
#                 cv2.rectangle(image, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), (0, 0, 0),
#                               thickness=cv2.FILLED)
#                 # 在真实框旁边添加类别标签
#                 cv2.putText(image, category, (xmin, ymin-5), cv2.FONT_ITALIC, 1.6, (255,255,255),2)
#
#             # 保存带有真实框和类别标签的图像
#             output_image_path = os.path.join(output_folder, os.path.basename(image_path))
#             cv2.imwrite(output_image_path, image)
#
# # 使用示例
# data_dir = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007' # 更改为您的VOC数据集路径
# save_dir = '/home/duomeitinrfx/data/tangka_magic_instrument/update/GT' # 更改为保存标注图像的路径
# os.makedirs(save_dir, exist_ok=True)
#
# draw_bounding_boxes_and_labels(data_dir, save_dir)


import xml.etree.ElementTree as ET
import cv2

def draw_bounding_boxes_and_label(image_path, xml_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        # 获取类别名称
        category = obj.find('name').text

        # 获取真实框坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 绘制矩形（真实框）
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

        # 文字背景
        (text_width, text_height), baseline = cv2.getTextSize(category, cv2.FONT_ITALIC,2, 2)
        cv2.rectangle(image, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), (0, 0, 0), thickness=cv2.FILLED)

        # 在真实框旁边添加类别标签
        cv2.putText(image, category, (xmin, ymin-5), cv2.FONT_ITALIC, 2, (255,255,255), 2)

    # 保存带有真实框和类别标签的图像
    cv2.imwrite(output_path, image)

# 使用示例
image_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/2756.jpg' # 更改为您的图像路径
xml_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations/2756.xml' # 更改为相应的 XML 标注文件路径
output_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/GT/2756.jpg' # 更改为保存标注图像的路径

draw_bounding_boxes_and_label(image_path, xml_path, output_path)

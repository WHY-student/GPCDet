import cv2
import xml.etree.ElementTree as ET
import os

# 设置图片和标签的路径
image_folder = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages'
annotation_folder = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/Annotations/'
save_folder = '/home/duomeitinrfx/data/tangka_magic_instrument/update/outall/'

# 确保保存图片的文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历标签文件夹中的所有XML文件
for filename in os.listdir(annotation_folder):
    if not filename.endswith('.xml'):
        continue

    # 解析XML文件
    tree = ET.parse(os.path.join(annotation_folder, filename))
    root = tree.getroot()

    # 获取图片路径
    image_file = root.find('filename').text
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # 检查图片是否正确加载
    if image is None:
        print(f"无法读取图像: {image_path}")
        continue

    # 遍历所有的目标对象
    for obj in root.iter('object'):
        # 获取边界框
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 绘制边界框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # 保存绘制了边界框的图像
    cv2.imwrite(os.path.join(save_folder, image_file), image)
    print(image_file+"成功")

print("处理完成。")

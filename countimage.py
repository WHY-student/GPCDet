import cv2
import os

# 数据集路径
dataset_dir = "/home/duomeitinrfx/data/tangka_magic_instrument/VOCdevkit/VOC2007/JPEGImages"

# 初始化变量
total_width = 0
total_height = 0
max_width = 0
max_height = 0
min_width = float('inf')
min_height = float('inf')
num_images = 0

# 遍历数据集中的每个图像
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_dir, filename)

        # 读取图像
        image = cv2.imread(image_path)

        # 获取图像尺寸
        height, width, _ = image.shape

        # 更新总和和计数
        total_width += width
        total_height += height
        num_images += 1

        # 更新最大尺寸和最小尺寸
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height

# 计算平均尺寸
avg_width = total_width / num_images
avg_height = total_height / num_images

# 打印结果
print("平均尺寸：{:.2f} x {:.2f}".format(avg_width, avg_height))
print("最大尺寸：{} x {}".format(max_width, max_height))
print("最小尺寸：{} x {}".format(min_width, min_height))

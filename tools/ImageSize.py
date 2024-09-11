from PIL import Image
import os
import matplotlib.pyplot as plt
import math

# 设置数据集目录
dataset_dir = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages'

# 存储图片尺寸的列表
widths = []
heights = []

# 遍历数据集目录中的图片文件
for filename in os.listdir(dataset_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # 打开图像文件
        with Image.open(os.path.join(dataset_dir, filename)) as img:
            # 获取图片尺寸
            width, height = img.size
            widths.append(width)
            heights.append(height)
plt.rcParams['font.size'] = 18
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, alpha=0.5)
plt.title('Image Sizes')
plt.xlabel('宽')
plt.ylabel('高')
plt.grid(True)

# 计算平均尺寸、最大尺寸和最小尺寸
average_width = math.fsum(widths) / len(widths)
average_height = math.fsum(heights) / len(heights)
max_width = max(widths)
max_height = max(heights)
min_width = min(widths)
min_height = min(heights)

print(f'Average Width: {average_width:.2f}')
print(f'Average Height: {average_height:.2f}')
print(f'Max Width: {max_width}')
print(f'Max Height: {max_height}')
print(f'Min Width: {min_width}')
print(f'Min Height: {min_height}')

# 显示散点图
plt.show()

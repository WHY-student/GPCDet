# import os
# import numpy as np
# import cv2
#
# files_dir = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/'
# files = os.listdir(files_dir)
#
# R = 0.
# G = 0.
# B = 0.
# R_2 = 0.
# G_2 = 0.
# B_2 = 0.
# N = 0
# i=0
# for file in files:
#     i+=1
#     if i%10==0:
#         print(i)
#     img = cv2.imread(files_dir+file)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.array(img)
#     h, w, c = img.shape
#     N += h*w
#
#     R_t = img[:, :, 0]
#     R += np.sum(R_t)
#     R_2 += np.sum(np.power(R_t, 2.0))
#
#     G_t = img[:, :, 1]
#     G += np.sum(G_t)
#     G_2 += np.sum(np.power(G_t, 2.0))
#
#     B_t = img[:, :, 2]
#     B += np.sum(B_t)
#     B_2 += np.sum(np.power(B_t, 2.0))
#
# R_mean = R/N
# G_mean = G/N
# B_mean = B/N
#
# R_std = np.sqrt(R_2/N - R_mean*R_mean)
# G_std = np.sqrt(G_2/N - G_mean*G_mean)
# B_std = np.sqrt(B_2/N - B_mean*B_mean)
#
# print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
# print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))

"""
计算训练集的三通道均值和标准差
适用于训练集中存在不同尺寸的图像
"""

from importlib.resources import path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from tqdm import trange

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os  # 用于读取文件夹内的文件列表


def calculate_average_histogram(image_paths):
    histogram_r = np.zeros(256)
    histogram_g = np.zeros(256)
    histogram_b = np.zeros(256)
    t=0;
    for path in image_paths:
        t+=1
        print(t)
        image = io.imread(path)
        for i in range(3):
            hist, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
            if i == 0:
                histogram_r += hist
            elif i == 1:
                histogram_g += hist
            else:
                histogram_b += hist

    histogram_r /= len(image_paths)
    histogram_g /= len(image_paths)
    histogram_b /= len(image_paths)

    return histogram_r, histogram_g, histogram_b


def plot_histograms(histogram_r, histogram_g, histogram_b, title):
    plt.figure(figsize=(10, 3))
    plt.plot(histogram_r, color='red', label='Red Channel')
    plt.plot(histogram_g, color='green', label='Green Channel')
    plt.plot(histogram_b, color='blue', label='Blue Channel')
    plt.title(title)
    plt.legend()
    plt.show()


# 假设你有两个文件夹分别包含唐卡图像和自然场景图像
thangka_images_folder = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/'
natural_images_folder = '/home/duomeitinrfx/data/VOC/VOCdevkit/VOC2007/JPEGImages/'

thangka_images_paths = [os.path.join(thangka_images_folder, f) for f in os.listdir(thangka_images_folder) if
                        f.endswith(('.png', '.jpg', '.jpeg'))]
natural_images_paths = [os.path.join(natural_images_folder, f) for f in os.listdir(natural_images_folder) if
                        f.endswith(('.png', '.jpg', '.jpeg'))]

# Calculate the average histograms
histogram_r_thangka, histogram_g_thangka, histogram_b_thangka = calculate_average_histogram(thangka_images_paths)
histogram_r_natural, histogram_g_natural, histogram_b_natural = calculate_average_histogram(natural_images_paths)

# Plot the histograms
plot_histograms(histogram_r_thangka, histogram_g_thangka, histogram_b_thangka,
                'Average Color Histogram for Thangka Images')
plot_histograms(histogram_r_natural, histogram_g_natural, histogram_b_natural,
                'Average Color Histogram for Natural Scene Images')




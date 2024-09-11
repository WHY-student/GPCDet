#获取图像HSV值
 
import cv2
import numpy as np

# IMG_SIZE=640
# sumH=0
# sumS=0
# sumV=0
# for i in range(0,3200):
#     img_array = cv2.imread("/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(i))
#     image = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))  #将图片统一为一样的大小
#     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     H, S, V = cv2.split(hsv)
#     #亮度（V）
#     v = V.ravel()[np.flatnonzero(V)]  #亮度非零的值
#     average_v  = sum(v)/len(v)  #计算亮度均值
#     sumV+=average_v
#     #饱和度（S）
#     s = S.ravel()[np.flatnonzero(S)]
#     average_s  = sum(s)/len(s)
#     sumS+=average_s
#     #色调（H）
#     h = H.ravel()[np.flatnonzero(H)]
#     average_h  = sum(h)/len(h)
#     sumH+=average_h
# print("平均色调: ",sumH/3200)
# print("平均饱和度: ",sumS/3200)
# print("平均亮度: ",sumV/3200)

img_array = cv2.imread(
    "/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(112))
image = cv2.resize(img_array, (640, 640))  # 将图片统一为一样的大小
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsv)
# 亮度（V）
v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
average_v = sum(v) / len(v)  # 计算亮度均值
print(average_v)
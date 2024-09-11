# import json
# import numpy as  np
# # 读取COCO格式的标签JSON文件
# with open('/home/duomeitinrfx/data/tangka_magic_instrument/update/coco_0.8_0.2/annotations/instances_train.json', 'r') as f:
#     data = json.load(f)
# # 创建一个16x16的矩阵用于存储类别组合的同时出现次数
# co_occurrence_matrix = np.zeros((16, 16), dtype=int)
#
# # 创建一个字典用于跟踪每张图片中出现的类别
# image_categories = {}
#
#
# # 遍历所有标注
# for annotation in data['annotations']:
#     image_id = annotation['image_id']
#     category_id = annotation['category_id']
#
#     if image_id not in image_categories:
#         image_categories[image_id] = set()
#
#     image_categories[image_id].add(category_id)
#
#
# #####计算每个类别出现次数
# M=np.zeros(16,dtype=int)
# #print(image_categories)
# # 遍历每张图片的类别，更新共现矩阵
# for categories in image_categories.values():
#     categories = list(categories)
#    # print(categories)
#     for i in range(len(categories)):
#         M[categories[i]-1]+=1
#         for j in range(i+1, len(categories)):
#             co_occurrence_matrix[categories[i] - 1, categories[j] - 1] += 1
#             co_occurrence_matrix[categories[j] - 1, categories[i] - 1] += 1
#
#
#
# # 打印16x16矩阵
# print(co_occurrence_matrix)
# print()
# print(M)
#
# #邻接矩阵
# M = M[:, np.newaxis]
# result = (co_occurrence_matrix / M.astype(float)).round(3)
# for i in range(len(result)):
#     print("[",end="")
#     for j in range(len(result[0])):
#         print(result[i][j],end=',')
#     print("]")


# import numpy as np
# import xlwt
# # 随机生成一个3×4的数组（值不超过10）
# # 创建excel表格类型文件
# python_list = co_occurrence_matrix.tolist()
# book = xlwt.Workbook(encoding='utf-8', style_compression=0)
# # 在excel表格类型文件中建立一张sheet表单
# sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
#
# for i in range(len(python_list)): #逐行
#     for j in range(len(python_list[0])): #逐列
#         sheet.write(i, j, python_list[i][j]) #将指定值写入第i行第j列
#
# save_path = 'data.xls'
# book.save(save_path)

import numpy as np
# import xlwt
# # 随机生成一个3×4的数组（值不超过10）
# # 创建excel表格类型文件
# python_list = result.tolist()
# book = xlwt.Workbook(encoding='utf-8', style_compression=0)
# # 在excel表格类型文件中建立一张sheet表单
# sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
#
# for i in range(len(python_list)): #逐行
#     for j in range(len(python_list[0])): #逐列
#         sheet.write(i, j, python_list[i][j]) #将指定值写入第i行第j列
#
# save_path = 'Adj.xls'
# book.save(save_path)

# np.array([[1.0,0.715,0.21,0.122,0.448,0.092,0.609,0.083,0.264,0.295,0.144,0.04,0.013,0.026,0.178,0.16],
#           [0.321,1.0,0.256,0.116,0.449,0.096,0.309,0.091,0.181,0.177,0.072,0.037,0.021,0.047,0.194,0.092],
#           [0.291,0.79,1.0,0.223,0.473,0.112,0.314,0.086,0.168,0.161,0.059,0.044,0.024,0.069,0.25,0.086],
#           [0.351,0.748,0.466,1.0,0.511,0.183,0.427,0.187,0.16,0.153,0.073,0.034,0.046,0.118,0.309,0.103],
#           [0.341,0.761,0.26,0.135,1.0,0.179,0.401,0.131,0.173,0.182,0.074,0.045,0.037,0.069,0.228,0.096],
#           [0.352,0.814,0.307,0.241,0.894,1.0,0.442,0.151,0.176,0.141,0.065,0.045,0.08,0.06,0.352,0.085],
#           [0.669,0.758,0.25,0.163,0.579,0.128,1.0,0.118,0.242,0.263,0.122,0.041,0.029,0.049,0.228,0.123],
#           [0.238,0.581,0.177,0.185,0.491,0.113,0.306,1.0,0.109,0.249,0.049,0.026,0.083,0.132,0.404,0.226],
#           [0.533,0.813,0.245,0.112,0.459,0.093,0.445,0.077,1.0,0.285,0.12,0.032,0.021,0.027,0.157,0.165],
#           [0.467,0.623,0.184,0.084,0.379,0.059,0.379,0.138,0.224,1.0,0.245,0.056,0.002,0.017,0.23,0.159],
#           [0.574,0.637,0.168,0.1,0.389,0.068,0.442,0.068,0.237,0.616,1.0,0.063,0.0,0.026,0.095,0.2],
#           [0.385,0.795,0.308,0.115,0.577,0.115,0.359,0.09,0.154,0.346,0.154,1.0,0.013,0.0,0.244,0.077],
#           [0.227,0.795,0.295,0.273,0.841,0.364,0.455,0.5,0.182,0.023,0.0,0.023,1.0,0.25,0.341,0.045],
#           [0.19,0.752,0.362,0.295,0.657,0.114,0.324,0.333,0.095,0.076,0.048,0.0,0.105,1.0,0.8,0.267],
#           [0.255,0.62,0.259,0.153,0.429,0.132,0.297,0.202,0.112,0.208,0.034,0.036,0.028,0.159,1.0,0.216],
#           [0.44,0.564,0.171,0.098,0.349,0.062,0.309,0.218,0.225,0.276,0.138,0.022,0.007,0.102,0.415,1.0]])


import json
import numpy as np

# 读取COCO格式的标签JSON文件
with open('/home/duomeitinrfx/data/tangka_magic_instrument/update/coco_0.8_0.2/annotations/instances_train.json', 'r') as f:
    data = json.load(f)

# 创建一个16x16的矩阵用于存储类别组合的同时出现次数
co_occurrence_matrix = np.zeros((16, 16), dtype=int)

# 创建一个字典用于跟踪每张图片中每个类别出现的次数
image_categories_count = {}

# 遍历所有标注
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id'] - 1  # 索引从0开始

    if image_id not in image_categories_count:
        image_categories_count[image_id] = np.zeros(16, dtype=int)

    image_categories_count[image_id][category_id] += 1

# 计算每个类别的总出现次数
category_totals = np.zeros(16, dtype=int)

# 更新共现矩阵
for counts in image_categories_count.values():
    for i in range(len(counts)):
        category_totals[i] += counts[i]
        for j in range(len(counts)):
            if i != j:  # 只在不同类别之间计算共现
                co_occurrence_matrix[i, j] += min(counts[i], counts[j])

# 打印共现矩阵
print("Co-occurrence Matrix:")
print(co_occurrence_matrix)
print()

# 计算条件概率矩阵
conditional_probability_matrix = np.zeros((16, 16), dtype=float)
for i in range(16):
    for j in range(16):
        if category_totals[i] > 0:
            conditional_probability_matrix[i, j] = co_occurrence_matrix[i, j] / category_totals[i]

conditional_probability_matrix = np.round(conditional_probability_matrix, 3)

# 打印条件概率矩阵
print("Conditional Probability Matrix:")
for row in conditional_probability_matrix:
    print("[", end="")
    for value in row:
        print(f"{value},", end="")
    print("]")

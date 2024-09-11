import json

# 定义函数来解析COCO格式的JSON注释文件并判断目标是否为小目标
def is_small_target_coco(json_file_path, threshold=0.1):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    small_target_count = 0
    large_target_count = 0

    # 遍历COCO数据集中的每一个注释
    for annotation in coco_data['annotations']:
        img_id = annotation['image_id']
        # 获取对应图像的信息
        img_info = next((item for item in coco_data['images'] if item["id"] == img_id), None)
        img_width = img_info['width']
        img_height = img_info['height']

        # 获取边界框信息
        bbox = annotation['bbox']
        # COCO的bbox格式是[xmin, ymin, width, height]
        target_width = bbox[2]
        target_height = bbox[3]

        # 计算目标的相对尺寸
        relative_width = target_width / img_width
        relative_height = target_height / img_height

        # 判断目标是否为小目标
        if relative_width <= threshold and relative_height <= threshold:
            small_target_count += 1
        else:
            large_target_count += 1

    # 计算占比
    total_target_count = small_target_count + large_target_count
    small_target_percentage = (small_target_count / total_target_count) * 100
    large_target_percentage = (large_target_count / total_target_count) * 100

    # 返回小目标和大目标的占比
    return small_target_percentage, large_target_percentage

# 指定COCO数据集的JSON文件路径
json_file_path = "/home/duomeitinrfx/data/coco/annotations/instances_train2017.json"

# 获取小目标和大目标的占比
small_target_percentage, large_target_percentage = is_small_target_coco(json_file_path, threshold=0.1)

# 打印结果
print("小目标占比: {:.2f}%".format(small_target_percentage))
print("大目标占比: {:.2f}%".format(large_target_percentage))

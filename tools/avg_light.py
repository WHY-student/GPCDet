from PIL import Image
import os

def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * -scale + index

    return brightness / scale

def average_brightness(folder_path):
    brightness_values = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            with Image.open(os.path.join(folder_path, image_file)) as img:
                brightness = calculate_brightness(img)
                brightness_values.append(brightness)

    if brightness_values:
        return sum(brightness_values) / len(brightness_values)
    else:
        return None

#平均亮度 127.5
# 使用示例
folder_path = '/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages'  # 将这里的路径替换为您的图片文件夹路径
print("Average Brightness:", average_brightness(folder_path))

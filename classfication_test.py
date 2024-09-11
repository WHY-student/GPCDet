import torch
from torchvision import transforms
from PIL import Image
from ResNet import ResNet, Bottleneck



# 图片路径
image_path = '/home/duomeitinrfx/data/tangka_magic_instrument/法器/16旗帜.png'

model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=16)

class_names=('Bowl', "Buddhist abbot's staff", 'Karma pestle', 'Peacock feather fan and mirror', 'Pipa', 'Precious mirror', 'Ruyi Bao', 'Scripture', 'Yak tail dusting', 'beads', 'canopy', 'flag', 'pagoda', 'sword', 'vajra Pestle', 'vajra bell')
# 调用测试函数
model.load_state_dict(torch.load('/home/duomeitinrfx/users/pengxl/mmdetection/best_model1024.pth'))
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
# 读取图像
image = Image.open(image_path)
# 预处理
images = transform(image)
images = images.unsqueeze(0)  # 添加 batch 维度
images=images.to(device)

# 设置模型为评估模式
model.eval()
model=model.to(device)
output = model(images)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
predicted_class = class_names[predicted_idx.item()]

print(f"Predicted class: {predicted_class}")
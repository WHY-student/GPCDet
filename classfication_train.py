import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from ResNet import ResNet, Bottleneck

# 定义数据路径
data_dir = '/home/duomeitinrfx/data/tangka_magic_instrument/update/classfication'

# 图像预处理和数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.435, 0.340, 0.283], [0.266, 0.231, 0.229])
    ]),
    'val': transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.435, 0.340, 0.283], [0.266, 0.231, 0.229])
    ]),
}

# 创建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
# 加载预训练的ResNet50模型
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=16)
# 修改模型的最后一层，适应分类任务
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
best_acc = 0.0
num_epochs = 100
for epoch in range(num_epochs):
    if epoch == 10:
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    elif epoch == 30:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

            if batch_idx % 20 == 0:
                print(f'{phase} Batch {batch_idx}/{len(dataloaders[phase]) - 1} Loss: {loss.item():.4f}')

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f'best_model1024.pth')
print("Training complete.")

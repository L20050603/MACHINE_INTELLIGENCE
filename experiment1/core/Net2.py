import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 创意组件：残差块 ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 核心：残差连接
        return F.relu(out)

# --- 主网络模型 ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 初始特征提取
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 卷积层级：逐渐增加通道数并缩小特征图
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        
        # 全连接输出
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局平均池化，增强空间鲁棒性
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)

# --- 训练逻辑包装类 ---
class MNISTExpert:
    def __init__(self, lr=0.001, data_dir="./data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.data_dir = data_dir
        
        # 数据增强 (对应你原代码中的 ImageAugmenter)
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def train_model(self, epochs=10, batch_size=64):
        train_loader = DataLoader(
            datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform),
            batch_size=batch_size, shuffle=True
        )
        
        # 学习率调度器：模拟你代码中的 lr *= 0.5
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                if batch_idx % 200 == 0:
                    print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')
            
            scheduler.step()
            print(f"--- Epoch {epoch} Avg Loss: {total_loss/len(train_loader):.4f} ---")

    def evaluate(self):
        test_loader = DataLoader(
            datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=1000
        )
        
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest Accuracy: {accuracy:.2f}%')
        return accuracy

    def save(self, path="mnist_model.pth"):
        torch.save(self.model.state_dict(), path)

# --- 使用示例 ---
if __name__ == "__main__":
    expert = MNISTExpert()
    expert.train_model(epochs=5) # 现代网络收敛极快，5轮即可
    expert.evaluate()

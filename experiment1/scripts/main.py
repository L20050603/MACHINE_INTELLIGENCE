import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core.Net import Net
from core.ImageAugmenter import ImageAugmenter
from core import MathTools
import torch
import torchvision
import torchvision.transforms as transforms
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

def load_mnist_data_pytorch(data_dir='./data'):
    """
    使用PyTorch下载MNIST数据集

    参数:
    data_dir: 数据保存目录

    返回:
    (x_train, y_train_onehot), (x_test, y_test_onehot)
    """
    print(f"正在下载/加载MNIST数据集到 {data_dir}...")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 定义数据转换：将PIL图像转换为Tensor，并归一化到[0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor并缩放到[0,1]
    ])

    # 下载训练集
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    # 下载测试集
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )

    print("数据集下载/加载完成！")

    # 转换为NumPy数组
    print("正在转换数据格式...")

    # 训练集
    x_train = []
    y_train = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        x_train.append(img.numpy().flatten())  # 将1x28x28展平为784
        y_train.append(label)

    # 测试集
    x_test = []
    y_test = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        x_test.append(img.numpy().flatten())
        y_test.append(label)

    # 转换为NumPy数组
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 转换为one-hot编码
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot

    y_train_onehot = to_one_hot(y_train)
    y_test_onehot = to_one_hot(y_test)

    print(f"数据转换完成！")
    print(f"训练集: {x_train.shape}, {y_train_onehot.shape}")
    print(f"测试集: {x_test.shape}, {y_test_onehot.shape}")

    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"  像素值范围: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  标签分布 - 训练集: {np.bincount(y_train)}")
    print(f"  标签分布 - 测试集: {np.bincount(y_test)}")

    # 显示文件信息
    print(f"\n数据文件位置:")
    print(f"  训练图像: {os.path.join(data_dir, 'MNIST/raw/train-images-idx3-ubyte')}")
    print(f"  训练标签: {os.path.join(data_dir, 'MNIST/raw/train-labels-idx1-ubyte')}")
    print(f"  测试图像: {os.path.join(data_dir, 'MNIST/raw/t10k-images-idx3-ubyte')}")
    print(f"  测试标签: {os.path.join(data_dir, 'MNIST/raw/t10k-labels-idx1-ubyte')}")

    return (x_train, y_train_onehot), (x_test, y_test_onehot)


def visualize_dataset_samples(x_train, y_train, num_samples=10):
    """
    可视化数据集样本
    """
    indices = np.random.choice(len(x_train), num_samples, replace=False)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {np.argmax(y_train[idx])}')
        plt.axis('off')

    plt.suptitle('MNIST数据集样本')
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, x_test, y_test, num_examples=5):
    """
    可视化预测结果
    """
    # 随机选择几个测试样本
    indices = np.random.choice(len(x_test), num_examples, replace=False)

    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        # 预测
        x = x_test[idx:idx + 1]
        y_pred = model.predict(x)
        pred_label = np.argmax(y_pred)
        true_label = np.argmax(y_test[idx])

        # 显示图像
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {y_pred[0][pred_label]:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 50)
    print("手写数字识别系统 (PyTorch版本)")
    print("=" * 50)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    data_dir = './data'  # 数据保存目录
    (x_train, y_train), (x_test, y_test) = load_mnist_data_pytorch(data_dir)

    # 显示数据样本
    print("\n显示数据集样本...")
    visualize_dataset_samples(x_train, y_train, num_samples=10)

    # 2. 创建模型
    print("\n[2] 创建神经网络...")
    model = Net(
        input_size=784,  # 28x28=784
        output_size=10,  # 10个数字
        linears=[128, 64]  # 两个隐藏层
    )
    print(f"网络结构: 784 -> 128 -> 64 -> 10")
    print(f"参数总数: {sum(l['W'].size + l['b'].size for l in model.layers)}")

    # 3. 训练模型
    print("\n[3] 开始训练...")
    history = model.train(
        x_train, y_train,
        batch_size=64,
        epochs=5,  # 实际训练可以用更多epochs
        lr=0.001,
        counts=200,  # 每200个batch打印一次
    )

    # 4. 评估模型
    print("\n[4] 评估模型...")
    train_acc = model.evaluate(x_train[:5000], y_train[:5000])  # 用部分训练数据评估
    test_acc = model.evaluate(x_test, y_test)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 5. 可视化结果
    print("\n[5] 可视化预测结果...")
    visualize_predictions(model, x_test, y_test)

    # 6. 保存模型
    print("\n[6] 保存模型...")
    model.save_model('mnist_model.npy')

    # 7. 演示数据增强
    print("\n[7] 数据增强效果演示...")
    augmenter = ImageAugmenter(image_shape=(28, 28))

    # 选择一个样本
    sample = x_test[0].reshape(28, 28)

    # 显示不同增强效果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    augmentations = [
        ('Original', sample),
        ('Translation', augmenter.random_translation(sample, max_shift=3)),
        ('Rotation', augmenter.random_rotation(sample, max_angle=30)),
        ('Scaling', augmenter.random_scaling(sample, (0.8, 1.2))),
        ('Noise', augmenter.add_noise(sample, 0.05)),
        ('Flip', augmenter.random_flip(sample, horizontal=True)),
        ('Erasing', augmenter.random_erasing(sample, 0.1)),
        ('Combined', augmenter.apply_augmentations(sample))
    ]

    for i, (title, img) in enumerate(augmentations):
        axes[i // 4, i % 4].imshow(img, cmap='gray')
        axes[i // 4, i % 4].set_title(title)
        axes[i // 4, i % 4].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n 程序运行完成！")

    # 显示数据文件夹内容
    print(f"\n数据文件夹 '{data_dir}' 内容:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # 只显示前5个文件
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... 还有 {len(files) - 5} 个文件")


if __name__ == "__main__":
    main()

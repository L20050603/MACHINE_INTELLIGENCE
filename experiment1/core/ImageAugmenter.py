import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom

class ImageAugmenter:
    """
    图像数据增强类

    支持的操作:
    - 随机平移
    - 随机旋转
    - 随机缩放
    - 随机翻转
    - 添加噪声
    - 随机擦除
    """

    def __init__(self, image_shape=(28, 28), random_seed=None):
        """
        初始化数据增强器

        参数:
        image_shape: 原始图像形状 (h, w)
        random_seed: 随机种子，用于重现结果
        """
        self.image_shape = image_shape
        if random_seed is not None:
            np.random.seed(random_seed)

        # 默认增强配置
        self.default_config = {
            'translation': {'enabled': True, 'max_shift': 2, 'probability': 0.5},
            'rotation': {'enabled': False, 'max_angle': 15, 'probability': 0.3},
            'scaling': {'enabled': False, 'scale_range': (0.9, 1.1), 'probability': 0.3},
            'flip': {'enabled': False, 'horizontal': True, 'vertical': False, 'probability': 0.5},
            'noise': {'enabled': False, 'noise_level': 0.01, 'probability': 0.3},
            'erasing': {'enabled': False, 'erase_ratio': 0.1, 'probability': 0.2}
        }

    def _reshape_to_image(self, batch):
        """将扁平化的数据重塑为图像格式"""
        if batch.ndim == 2:  # (batch_size, flattened)
            return batch.reshape(-1, self.image_shape[0], self.image_shape[1])
        elif batch.ndim == 3:  # (batch_size, h, w)
            if batch.shape[1:] != self.image_shape:
                raise ValueError(f"图像形状不匹配，期望 {self.image_shape}，得到 {batch.shape[1:]}")
            return batch
        else:
            raise ValueError(f"不支持的输入维度: {batch.ndim}")

    def _flatten_batch(self, images_3d):
        """将图像格式的数据扁平化"""
        return images_3d.reshape(images_3d.shape[0], -1)

    def random_translation(self, image, max_shift=2):
        """
        随机平移

        参数:
        image: 单张图像 (h, w)
        max_shift: 最大平移像素数

        返回:
        平移后的图像
        """
        dy, dx = np.random.randint(-max_shift, max_shift + 1, size=2)

        # 使用np.roll进行循环平移
        shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))

        # 将滚回来的边缘清零
        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0
        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0

        return shifted

    def random_rotation(self, image, max_angle=15):
        """
        随机旋转

        参数:
        image: 单张图像 (h, w)
        max_angle: 最大旋转角度

        返回:
        旋转后的图像
        """
        angle = np.random.uniform(-max_angle, max_angle)
        #边界常量填充为黑色(cval=0)
        rotated = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
        return rotated

    def random_scaling(self, image, scale_range=(0.9, 1.1)):
        """
        随机缩放

        参数:
        image: 单张图像 (h, w)
        scale_range: 缩放因子范围 (min, max)

        返回:
        缩放后的图像
        """
        scale = np.random.uniform(*scale_range)

        if scale != 1.0:
            h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)

            # 缩放图像
            scaled = zoom(image, scale, order=1)

            # 裁剪或填充回原大小
            if scale > 1.0:  # 放大后裁剪
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                result = scaled[start_h:start_h + h, start_w:start_w + w]
            else:  # 缩小后填充
                result = np.zeros_like(image)
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                result[start_h:start_h + new_h, start_w:start_w + new_w] = scaled
        else:
            result = image.copy()

        return result

    def random_scaling(self, image, scale_range=(0.9, 1.1)):
        """Scale an image and center-crop/pad back to the original shape."""
        scale = np.random.uniform(*scale_range)
        if scale == 1.0:
            return image.copy()

        h, w = image.shape
        scaled = zoom(image, scale, order=1)
        sh, sw = scaled.shape

        result = np.zeros_like(image)
        src_y0 = max(0, (sh - h) // 2)
        src_x0 = max(0, (sw - w) // 2)
        src_y1 = min(sh, src_y0 + h)
        src_x1 = min(sw, src_x0 + w)

        crop = scaled[src_y0:src_y1, src_x0:src_x1]
        ch, cw = crop.shape
        dst_y0 = max(0, (h - ch) // 2)
        dst_x0 = max(0, (w - cw) // 2)
        result[dst_y0:dst_y0 + ch, dst_x0:dst_x0 + cw] = crop
        return result

    def random_flip(self, image, horizontal=True, vertical=False):
        """
        随机翻转

        参数:
        image: 单张图像 (h, w)
        horizontal: 是否允许水平翻转
        vertical: 是否允许垂直翻转

        返回:
        翻转后的图像
        """
        result = image.copy()

        if horizontal and np.random.random() > 0.5:
            result = np.fliplr(result)
        if vertical and np.random.random() > 0.5:
            result = np.flipud(result)

        return result

    def add_noise(self, image, noise_level=0.01):
        """
        添加高斯噪声

        参数:
        image: 单张图像 (h, w)
        noise_level: 噪声水平（标准差）

        返回:
        添加噪声后的图像
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)  # 假设图像在[0,1]范围内

    def random_erasing(self, image, erase_ratio=0.1):
        """
        随机擦除

        参数:
        image: 单张图像 (h, w)
        erase_ratio: 擦除区域占图像的比例

        返回:
        擦除后的图像
        """
        h, w = image.shape
        erase_h = int(np.sqrt(erase_ratio * h * w))
        erase_w = erase_h

        if erase_h < h and erase_w < w:
            x = np.random.randint(0, w - erase_w)
            y = np.random.randint(0, h - erase_h)

            result = image.copy()
            result[y:y + erase_h, x:x + erase_w] = 0  # 用0填充擦除区域
            return result

        return image.copy()

    def apply_augmentations(self, image, config=None):
        """
        应用一系列数据增强

        参数:
        image: 单张图像 (h, w)
        config: 增强配置，None表示使用默认配置

        返回:
        增强后的图像
        """
        if config is None:
            config = self.default_config

        augmented = image.copy()

        # 按照概率应用各种增强
        if config['translation']['enabled'] and np.random.random() < config['translation']['probability']:
            augmented = self.random_translation(augmented, config['translation']['max_shift'])

        if config['rotation']['enabled'] and np.random.random() < config['rotation']['probability']:
            augmented = self.random_rotation(augmented, config['rotation']['max_angle'])

        if config['scaling']['enabled'] and np.random.random() < config['scaling']['probability']:
            augmented = self.random_scaling(augmented, config['scaling']['scale_range'])

        if config['flip']['enabled'] and np.random.random() < config['flip']['probability']:
            augmented = self.random_flip(augmented,
                                         horizontal=config['flip']['horizontal'],
                                         vertical=config['flip']['vertical'])

        if config['noise']['enabled'] and np.random.random() < config['noise']['probability']:
            augmented = self.add_noise(augmented, config['noise']['noise_level'])

        if config['erasing']['enabled'] and np.random.random() < config['erasing']['probability']:
            augmented = self.random_erasing(augmented, config['erasing']['erase_ratio'])

        return augmented

    def augment_batch(self, images, config=None, return_flat=True):
        """
        对batch进行数据增强

        参数:
        images: 输入图像batch，形状可以是(batch_size, 784)或(batch_size, 28, 28)
        config: 增强配置
        return_flat: 是否返回扁平化格式(batch_size, 784)

        返回:
        增强后的图像batch
        """
        # 重塑为3D格式
        imgs_3d = self._reshape_to_image(images)
        batch_size = imgs_3d.shape[0]

        # 创建输出数组
        augmented_3d = np.zeros_like(imgs_3d)

        # 对每张图像应用增强
        for i in range(batch_size):
            augmented_3d[i] = self.apply_augmentations(imgs_3d[i], config)

        # 根据要求返回格式
        if return_flat and images.ndim == 2:
            return self._flatten_batch(augmented_3d)
        else:
            return augmented_3d

    def set_config(self, new_config):
        """更新增强配置"""
        self.default_config.update(new_config)

    def get_config(self):
        """获取当前配置"""
        return self.default_config.copy()

    def visualize_augmentations(self, image, num_examples=5, save_path=None):
        """
        可视化多种增强效果

        参数:
        image: 单张图像 (28, 28) 或扁平化的(784,)
        num_examples: 生成几个增强例子
        save_path: 保存图像的路径（可选）
        """

        # 确保是2D图像
        if image.ndim == 1:
            image = image.reshape(self.image_shape)

        fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))

        # 第一行显示不同的单一增强
        augmentations = [
            ('Original', lambda x: x),
            ('Translation', lambda x: self.random_translation(x, max_shift=3)),
            ('Rotation', lambda x: self.random_rotation(x, max_angle=30)),
            ('Scaling', lambda x: self.random_scaling(x, (0.8, 1.2))),
            ('Noise', lambda x: self.add_noise(x, 0.05))
        ]

        for i, (name, aug_func) in enumerate(augmentations[:num_examples]):
            axes[0, i].imshow(aug_func(image), cmap='gray')
            axes[0, i].set_title(name)
            axes[0, i].axis('off')

        # 第二行显示组合增强的随机结果
        for i in range(num_examples):
            augmented = self.apply_augmentations(image)
            axes[1, i].imshow(augmented, cmap='gray')
            axes[1, i].set_title(f'Combined {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

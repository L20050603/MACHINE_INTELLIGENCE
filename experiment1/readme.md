# 基于 BP 算法的手写数字识别综合设计实验

本项目以经典 MNIST 手写数字识别为任务主线，实现了一个从算法训练到前端交互演示的完整实验系统。

核心分类器是 **NumPy 手写 BP 神经网络**，没有直接使用 PyTorch 的自动求导来完成主体分类模型。在此基础上，项目加入了两类自主创新：

- **强化学习式数据增强选择**：使用 epsilon-greedy 多臂老虎机自动选择更有效的数据增强策略。
- **ACGAN 生成式样本增强**：训练带类别条件的生成器，生成指定数字的手写样本，并用于扩充 BP 训练集。

项目还提供了本地 Web 前端：可以在浏览器中手写数字并调用后端模型预测；右侧小窗口可以展示 ACGAN 生成样本。

## 项目特点

- NumPy 手写多层 BP 网络：前向传播、Softmax、交叉熵、反向传播、Adam 更新、Dropout。
- 使用 `torchvision.datasets.MNIST` 加载 MNIST 数据集，无需手动准备 idx 文件。
- 提供图像增强：平移、旋转、缩放、噪声、遮挡。
- 使用 Multi-Armed Bandit 思想，自适应选择增强策略。
- 使用 ACGAN 生成带标签的合成样本，提升训练集多样性。
- 提供 PyTorch CNN/Residual 网络作为对照实验。
- 提供前端画板、预测概率条、Top 3、预处理预览和 ACGAN 生成窗口。

## 目录结构

```text
MACHINE_INTELLIGENCE/
  readme.md
  requirements.txt
  experiment1/
    web_app.py                    本地 Web 服务和 API 入口
    core/                         核心算法与模型
      Net.py                      NumPy BP 神经网络
      MathTools.py                Softmax、ReLU、交叉熵等基础函数
      ImageAugmenter.py           数据增强工具
      rl_augmenter.py             强化学习式增强策略选择
      acgan_synthesizer.py        ACGAN 生成器与判别器
      mnist_data.py               MNIST 加载与 one-hot 转换
      mnist_downloader.py         MNIST 下载兜底工具
      Net2.py                     PyTorch CNN/Residual 对照模型
    scripts/                      训练脚本
      train_baseline_bp.py        原始 BP（无增强）
      train_bp.py                 增强版 BP（归一化 + Dropout + Adam）
      train_rl_bp.py              强化学习增强版 BP
      train_gan_augmented_bp.py   ACGAN 生成增强 + BP 训练
      train_cnn.py                CNN/Residual 对照模型
      _train_rl_gpu_100ep.py      GPU 加速 100epoch RL 训练（含对比图）
      generate_plots.py           生成实验对比图
    web/                          前端页面
      index.html
      styles.css
      app.js
    artifacts/                    训练产物目录（运行后自动生成）
      models/                     模型权重（前端加载）
      acgan/                      ACGAN 权重、样本图、loss 曲线
        samples_100epoch/         每 epoch 生成样本图
      experiments/                实验结果
        20epoch/                  5 模型 × 20 轮实验
        100epoch/                 5 模型 × 100 轮实验
          comparison/             准确率对比图
```

## 环境准备

建议先创建并激活一个 Python 环境，例如：

```bash
conda create -n study python=3.11
conda activate study
```

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `torchvision`

说明：如果你已经有可用环境，只需要在仓库根目录执行 `pip install -r requirements.txt` 即可。

## 方法一：训练增强版BP

如果只想训练增强版 BP 网络（归一化 + Dropout + Adam），可以运行：

```bash
cd experiment1
python scripts/train_bp.py --epochs 10
```

如果想完全关闭数据增强，只保留 MLP + Dropout + Adam：

```bash
cd experiment1
python scripts/train_bp.py --epochs 10 --no-augmentation
```

推荐用于正式实验的配置：

```bash
cd experiment1
python scripts/train_bp.py --epochs 100 --hidden-layers 256,128,64 --dropout 0.1 --lr 0.001
```

如果希望后期更细致收敛，可以加入学习率衰减：

```bash
cd experiment1
python scripts/train_bp.py --epochs 120 --hidden-layers 512,256,128 --dropout 0.1 --lr 0.001 --lr-decay-step 50 --lr-decay-gamma 0.5
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--epochs` | `10` | 训练轮数。快速验证可以用 10，正式实验建议 50-120。 |
| `--batch-size` | `64` | mini-batch 大小。一般 64 或 128 都可以。 |
| `--lr` | `0.001` | Adam 学习率。过大可能震荡，过小收敛慢。 |
| `--hidden-layers` | `256,128,64` | MLP 隐藏层结构，例如 `512,256,128` 表示三层隐藏层。 |
| `--dropout` | `0.1` | Dropout 比例。常用 `0.1` 到 `0.3`。 |
| `--no-augmentation` | 关闭状态 | 加上后不使用数据增强，只训练纯 MLP。 |
| `--lr-decay-step` | `0` | 学习率衰减间隔。`0` 表示不衰减。 |
| `--lr-decay-gamma` | `0.5` | 每次衰减时学习率乘以该系数。 |
| `--val-size` | `5000` | 从训练集中划分多少样本作为验证集。 |
| `--limit-train` | 无 | 限制训练集大小，调试时使用。 |
| `--limit-test` | 无 | 限制测试集大小，调试时使用。 |
| `--save-path` | `artifacts/models/enhanced_bp.npy` | 模型保存路径。 |

默认输出：

```text
experiment1/artifacts/models/enhanced_bp.npy
```

这个脚本适合作为 baseline，对比各类增强方法。

## 方法二：训练强化学习增强版BP

强化学习部分使用 **Multi-Armed Bandit** 中的 **epsilon-greedy** 策略。

在本项目中：

- 一个动作表示一种数据增强策略。
- 奖励表示当前 epoch 后验证集准确率的提升。
- agent 在“探索新策略”和“利用当前最佳策略”之间平衡。

可选增强动作包括：

```text
clean
shift
rotate
scale
noise
erase
mixed_light
mixed_strong
```

运行训练：

```bash
cd experiment1
python scripts/train_rl_bp.py --epochs 5
```

快速调试版：

```bash
cd experiment1
python scripts/train_rl_bp.py --limit-train 12000 --epochs 3
```

正式实验建议：

```bash
cd experiment1
python scripts/train_rl_bp.py --epochs 50
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--epochs` | `5` | 强化学习增强版BP 的训练轮数。 |
| `--batch-size` | `64` | mini-batch 大小。 |
| `--lr` | `0.001` | BP 网络学习率。 |
| `--val-size` | `5000` | 验证集大小，reward 根据验证集准确率变化计算。 |
| `--limit-train` | 无 | 限制训练集大小，用于快速调试。 |
| `--limit-test` | 无 | 限制测试集大小，用于快速调试。 |
| `--save-path` | `artifacts/models/rl_enhanced_bp.npy` | 强化学习增强版BP 权重保存路径。 |
| `--history-path` | `artifacts/rl_augmentation_history.json` | 保存每轮策略选择、loss、reward、验证准确率。 |
| `--output-dir` | 无 | 指定后将模型、历史记录和图表一起保存到该目录。 |

强化学习内部参数目前写在 `core/rl_augmenter.py` 中：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `epsilon` | `0.25` | 初始探索概率。 |
| `decay` | `0.96` | 每轮后 epsilon 衰减倍率。 |
| `min_epsilon` | `0.05` | 最低探索概率。 |
| `reward` | `(当前验证准确率 - 最佳验证准确率) × 100` | 百分比提升作为奖励信号。 |

默认输出：

```text
experiment1/artifacts/models/rl_enhanced_bp.npy
experiment1/artifacts/rl_augmentation_history.json
```

其中：

- `mnist_model.npy` 是 BP 网络权重，前端预测会自动加载它。
- `rl_augmentation_history.json` 记录每个 epoch 选择的增强策略、loss、验证集准确率和 reward。

## 方法三：训练 ACGAN + 生成样本增强 BP

ACGAN 是 Auxiliary Classifier GAN，即带辅助分类器的生成对抗网络。

本项目中的设计：

- Generator 输入随机噪声和数字标签，生成指定类别的手写数字。
- Discriminator 同时判断图片真假，并预测数字类别。
- 生成样本天然带标签，可以直接拼接到 BP 网络训练集中。

运行：

```bash
cd experiment1
python scripts/train_gan_augmented_bp.py --gan-epochs 3 --bp-epochs 5 --synthetic-per-class 300
```

如果还没有生成器权重，脚本会先训练 ACGAN，再生成合成样本，最后训练 BP 模型。

默认输出：

```text
experiment1/artifacts/acgan/acgan_mnist.pth
experiment1/artifacts/acgan/samples_100epoch/acgan_epoch_01.png
experiment1/artifacts/models/acgan_enhanced_bp.npy
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--gan-epochs` | `3` | ACGAN 生成器训练轮数。生成器想要更好效果通常需要更多轮。 |
| `--bp-epochs` | `5` | 使用真实样本 + 生成样本训练 BP 的轮数。 |
| `--batch-size` | `64` | BP 训练 batch 大小。 |
| `--lr` | `0.001` | BP 网络学习率。 |
| `--synthetic-per-class` | `300` | 每个数字类别生成多少张合成样本。 |
| `--synthetic-min-confidence` | `0.0` | 使用 ACGAN 判别器的辅助分类头过滤生成样本。比如 `0.7` 表示只保留判别器认为属于目标类别且置信度不低于 0.7 的样本。 |
| `--retrain-gan` | 关闭状态 | 即使已有 `acgan_mnist.pth`，也强制重新训练生成器。 |
| `--limit-train` | 无 | 限制真实训练集大小，便于快速调试。 |
| `--data-dir` | `experiment1/data` | MNIST 数据目录。 |
| `--gan-dir` | `artifacts/acgan` | ACGAN 权重与样例图保存目录。 |
| `--save-path` | `artifacts/models/acgan_enhanced_bp.npy` | ACGAN增强版BP 模型保存路径。 |

快速调试示例：

```bash
cd experiment1
python scripts/train_gan_augmented_bp.py --limit-train 10000 --gan-epochs 1 --bp-epochs 2 --synthetic-per-class 50
```

注意：如果 `artifacts/acgan/acgan_mnist.pth` 已经存在，脚本默认会复用已有生成器，不会重复训练 ACGAN。如果想重新训练生成器，可以加 `--retrain-gan`。

如果发现“选择数字 8 却生成像 7 的样本”，说明 ACGAN 的条件生成质量还不够稳定。可以尝试：

```bash
cd experiment1
python scripts/train_gan_augmented_bp.py --retrain-gan --gan-epochs 50 --bp-epochs 80 --synthetic-per-class 500 --synthetic-min-confidence 0.7
```

前端显示”ACGAN 已加载”只表示找到了生成器权重，不代表生成质量一定好。建议观察 `artifacts/acgan/samples_100epoch/acgan_epoch_*.png`，现在样例图按”每行一个数字类别”保存，更容易检查各类别质量。

## 方法四：训练 CNN/Residual 对照模型

CNN 也是基于反向传播训练的神经网络，只是这里使用 PyTorch 自动求导完成梯度计算，而不是像 `Net.py` 那样手写反向传播。

运行：

```bash
cd experiment1
python scripts/train_cnn.py --epochs 5
```

默认输出：

```text
experiment1/artifacts/models/cnn_residual.pth
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--epochs` | `5` | CNN 训练轮数。CNN 收敛通常比 MLP 快。 |
| `--batch-size` | `64` | batch 大小。 |
| `--lr` | `0.001` | AdamW 学习率。 |
| `--data-dir` | `experiment1/data` | MNIST 数据目录。 |
| `--save-path` | `artifacts/models/cnn_residual.pth` | CNN 权重保存路径。 |

## 前端演示

启动本地服务：

```bash
cd experiment1
python web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

页面功能：

- 在左侧画板手写数字。
- 在”模型权重”下拉框中选择 BP、增强版BP、强化学习增强版BP、ACGAN增强版BP 或 CNN/Residual BP。
- 点击”识别”，后端调用对应模型预测。
- 右侧显示预测结果、置信度、10 类概率条和 Top 3。
- 预处理窗口显示浏览器端缩放到 28x28 后的输入。
- 右侧底部 ACGAN 小窗口可以选择数字并生成合成样本。

前端模型加载规则：

- 增强版BP：`experiment1/artifacts/models/enhanced_bp.npy`
- 强化学习增强版BP：`experiment1/artifacts/models/rl_enhanced_bp.npy`
- ACGAN增强版BP：`experiment1/artifacts/models/acgan_enhanced_bp.npy`
- CNN/Residual BP：`experiment1/artifacts/models/cnn_residual.pth`
- ACGAN 生成器默认查找：`experiment1/artifacts/acgan/acgan_mnist.pth`

如果页面显示“未训练”，说明对应权重文件还没有生成。先运行训练脚本即可。

## 常用命令速查

安装依赖：

```bash
pip install -r requirements.txt
```

训练 RL + BP：

```bash
cd experiment1
python scripts/train_rl_bp.py --epochs 5
```

训练增强版BP：

```bash
cd experiment1
python scripts/train_bp.py --epochs 10
```

训练 ACGAN增强版BP：

```bash
cd experiment1
python scripts/train_gan_augmented_bp.py --gan-epochs 3 --bp-epochs 5
```

训练 CNN 对照模型：

```bash
cd experiment1
python scripts/train_cnn.py --epochs 5
```

启动前端：

```bash
cd experiment1
python web_app.py
```

查看脚本参数：

```bash
cd experiment1
python scripts/train_rl_bp.py --help
python scripts/train_gan_augmented_bp.py --help
```

## 实验报告可写思路

可以按以下结构撰写实验报告：

1. 实验目的：掌握 BP 神经网络训练流程，并完成手写数字识别系统。
2. 数据集介绍：MNIST，28x28 灰度图，10 个数字类别。
3. BP 网络设计：输入层 784 维，隐藏层 `[128, 64]`，输出层 10 类。
4. 训练算法：前向传播、交叉熵损失、反向传播、Adam 优化。
5. 基础增强：平移、旋转、缩放、噪声、遮挡。
6. 创新点一：强化学习 bandit 自动选择增强策略。
7. 创新点二：ACGAN 生成带标签样本，扩充训练集。
8. 前端系统：画板输入、28x28 预处理、后端预测、结果可视化。
9. 对照实验：BP、增强版BP、强化学习增强版BP、ACGAN增强版BP、CNN/Residual BP。
10. 结果分析：准确率、loss 曲线、生成样本质量、不同增强方式的影响。

## 常见问题

**1. 为什么运行前端后显示”未训练”？**

说明还没有生成任何可用模型权重。先运行一个训练脚本，例如：

```bash
cd experiment1
python scripts/train_rl_bp.py --epochs 5
```

训练后的主要权重会保存到 `experiment1/artifacts/models/`。

**2. 为什么 ACGAN 小窗口显示”未训练”？**

说明还没有生成 `experiment1/artifacts/acgan/acgan_mnist.pth`。运行：

```bash
cd experiment1
python scripts/train_gan_augmented_bp.py --gan-epochs 3 --bp-epochs 5
```

**3. 第一次运行为什么会比较慢？**

第一次会通过 `torchvision` 下载 MNIST 数据集，并且 ACGAN 训练本身比普通 BP 更耗时。

**4. MNIST 下载时出现 SSL 或 404 错误怎么办？**

项目已经加入备用下载逻辑。脚本会先尝试 `torchvision.datasets.MNIST(download=True)`，如果官方镜像或 SSL 连接失败，会自动从备用镜像补齐缺失的 MNIST gzip 文件并解压。通常只需要重新运行原训练命令。

**5. 如果没有 conda，可以直接用普通 Python 吗？**

可以。只要当前 Python 环境安装了 `requirements.txt` 中的依赖，就可以直接使用 `python ...` 命令运行。

## 产物说明

训练产生的模型和中间结果都放在 `experiment1/artifacts/` 下，并已在 `.gitignore` 中忽略，避免把大文件提交到仓库。

# 基于 BP 算法的手写数字识别综合实验

本项目围绕 MNIST 手写数字识别完成综合设计实验：核心模型使用 NumPy 手写 BP 神经网络，包含前向传播、Softmax、交叉熵、反向传播、Adam 参数更新、Dropout 与数据增强；同时提供 PyTorch 残差 CNN 作为对照实验。

## 自主创新点

新增 `experiment1/rl_augmenter.py`：把数据增强策略建模为强化学习中的动作，使用 epsilon-greedy 多臂老虎机根据验证集准确率提升来更新策略价值。可选择的动作包括 clean、shift、rotate、scale、noise、erase、mixed_light、mixed_strong。这样可以展示“模型训练 + 策略自适应”的设计，而不是固定增强参数。

## 前端

新增轻量前端与 API：

- `experiment1/web/index.html`：手写数字画板、预处理预览、Top 3 结果、概率条。
- `experiment1/web_app.py`：本地 HTTP 服务，加载 `mnist_model.npy` 后提供 `/api/predict`。

运行方式：

```bash
cd experiment1
E:\Conda\envs\study\python.exe web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果页面显示“未训练”，说明还没有找到 `mnist_model.npy`。先训练并保存模型，或把已有权重放到项目根目录或 `experiment1/` 下。

## 建议实验结构

1. 基础 BP：`Net.py` 完成多层全连接网络，输入 784 维，输出 10 类。
2. 数据增强：`ImageAugmenter.py` 提供平移、旋转、缩放、噪声、遮挡等方法。
3. 创新训练：`rl_augmenter.py` 用强化学习式 bandit 自动选择增强策略。
4. 对照模型：`Net2.py` 使用 PyTorch CNN/Residual Block 作为性能上界参考。
5. 前端演示：手写输入经过浏览器端 28x28 归一化，再调用 BP 模型预测。

## 依赖

```bash
E:\Conda\envs\study\python.exe -m pip install -r requirements.txt
```

如果已经激活环境，也可以使用：

```bash
conda activate study
cd experiment1
python web_app.py
```

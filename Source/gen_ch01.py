#!/usr/bin/env python3
"""Generate Jupyter notebooks for chapters 1-7 of 李宏毅深度学习教程"""
import json, os

NB_DIR = r"D:\AI_main\mynnew_code\pytorch_go\Source\notebooks"
os.makedirs(NB_DIR, exist_ok=True)

def make_notebook(cells, filepath):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": []
    }
    for cell_type, source in cells:
        if isinstance(source, list):
            source = "\n".join(source)
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": source.split("\n") if cell_type == "code" else source,
            "outputs": [],
            "execution_count": None
        } if cell_type == "code" else {
            "cell_type": cell_type,
            "metadata": {},
            "source": source
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {filepath}")

# ========================== CHAPTER 1 ==========================
ch01_cells = []

ch01_cells.append(("markdown", """# 第1章：机器学习基础

## 知识地图

```
第1章 机器学习基础
├── 1.1 案例学习：视频观看次数预测
│   ├── 机器学习的三个步骤
│   │   ├── 步骤1：写出带未知参数的函数（模型）
│   │   ├── 步骤2：定义损失函数
│   │   └── 步骤3：最优化（梯度下降）
│   ├── 梯度下降的详细过程
│   │   ├── 单参数梯度下降
│   │   ├── 多参数梯度下降
│   │   ├── 局部最小值与全局最小值
│   │   └── 超参数：学习率
│   └── 训练实战与结果分析
├── 1.2 线性模型
│   ├── 1.2.1 分段线性曲线
│   │   ├── 线性模型的局限性
│   │   ├── Hard Sigmoid 函数
│   │   ├── Sigmoid 函数逼近
│   │   ├── 多特征输入
│   │   ├── 矩阵表示与向量化
│   │   └── 批量（Batch）与回合（Epoch）
│   ├── 1.2.2 模型变形
│   │   ├── ReLU 激活函数
│   │   ├── Sigmoid vs ReLU
│   │   ├── 神经网络（多层结构）
│   │   ├── 深度学习的历史
│   │   └── 过拟合现象
│   └── 1.2.3 机器学习框架
│       ├── 训练集与测试集
│       ├── 训练三步曲总结
│       └── Kaggle提交流程
└── 关键概念连接
```

本章是全书的基础，建立了机器学习的核心框架。"""))

ch01_cells.append(("markdown", """## 1. 机器学习是什么？

### 核心直觉

机器学习，顾名思义，就是让**机器具备学习的能力**。更具体地说，机器学习就是让机器具备**找到一个函数**的能力。

用一个类比来理解：假设你想教会一个小孩识别猫。你不会给小孩写一个详细的规则手册（"猫有两只尖耳朵、有胡须、会喵喵叫..."），而是会给他看很多猫的图片，小孩自己从图片中"学习"到什么特征能识别猫。机器学习做的就是这样的事情——给机器大量数据，让它自己找出输入和输出之间的函数关系。

### 为什么需要机器学习？

某些问题太复杂，人类无法手工编写规则：
- **语音识别**：输入声音信号 → 输出文字。这个函数极其复杂，没人能手工写出来
- **图像识别**：输入图片 → 输出图片内容
- **AlphaGo下围棋**：输入棋盘状态 → 输出下一步落子位置

### 机器学习的三种任务类型

| 任务类型 | 输出形式 | 举例 |
|---------|---------|------|
| **回归（Regression）** | 一个数值（标量） | 预测明天PM2.5的数值 |
| **分类（Classification）** | 从预设选项中选择一个 | 垃圾邮件检测、AlphaGo落子（19×19选1） |
| **结构化学习（Structured Learning）** | 有结构的物体 | 让机器画一幅画、写一篇文章 |

> **关键理解**：AlphaGo表面上是在下围棋，但实际上它是一个**分类问题**。棋盘有19×19=361个位置，AlphaGo要做的是从361个选项中选出下一步落子位置。"""))

ch01_cells.append(("markdown", """## 2. 案例学习：视频观看次数预测

本节通过一个完整的案例——预测YouTube频道的每日观看次数——来展示机器学习的三个步骤。

### 问题设定

- **输入（特征）**：频道过去的信息（点赞数、订阅数、历史观看次数等）
- **输出（预测）**：明天的总观看次数
- **数据**：频道从2017年1月1日到2020年12月31日的每日观看次数

### 步骤1：写出带未知参数的函数（建立模型）

最简单的猜测：今天的观看次数和昨天的观看次数有关联。

$$y = b + w \\cdot x_1$$

其中：
- $y$：预测值（今天观看次数）
- $x_1$：特征（昨天观看次数）
- $w$：**权重（weight）**，未知参数，需要通过数据学习
- $b$：**偏置（bias）**，未知参数，需要通过数据学习

这个带有未知参数的函数被称为**模型（model）**。这里的$x_1$是**特征（feature）**，$w$和$b$是**参数（parameter）**。

> **领域知识的作用**：为什么猜测是$y = b + wx_1$这种形式？这来自对问题的理解——“今天的观看次数大概率和昨天的差不多”。这就是**领域知识（domain knowledge）**的价值。"""))

ch01_cells.append(("markdown", """### 步骤2：定义损失函数（Loss Function）

损失函数$L(b, w)$用于评估：给定一组参数$(b, w)$，模型的预测效果有多好（或多差）。

#### 损失的计算过程

1. 设定参数值，例如 $b = 500, w = 1$，模型变为 $y = 500 + x_1$
2. 用训练数据计算预测值：如果1月1日观看次数$x_1 = 4800$，则预测值$\\hat{y} = 500 + 4800 = 5300$
3. 对标真实值（标签$y = 4900$），计算误差 $e_1 = |y - \\hat{y}| = |4900 - 5300| = 400$
4. 对所有训练数据（3年，约1095天）重复此过程
5. 将所有误差求和取平均：

$$L = \\frac{1}{N}\\sum_{n=1}^{N} e_n$$

其中$N$是训练数据的总数量。$L$越大参数越差，$L$越小参数越好。

#### 常用的误差计算方法

- **平均绝对误差（MAE）**：$e = |\\hat{y} - y|$
- **均方误差（MSE）**：$e = (\\hat{y} - y)^2$
- **交叉熵（Cross Entropy）**：当$y$和$\\hat{y}$都是概率分布时使用

> **重要认知**：损失函数是自己定义的！你可以根据任务特性选择不同的损失函数。MAE对异常值不敏感，MSE对大的误差惩罚更重。

#### 误差表面（Error Surface）

当尝试不同的$w$和$b$组合并计算对应的损失后，可以画出**等高线图**，这就是**误差表面**。越偏红色的地方损失越大，越偏蓝色损失越小。优化的目标就是找到误差表面上"最蓝"的那个点。"""))

ch01_cells.append(("markdown", """### 步骤3：最优化——梯度下降（Gradient Descent）

#### 为什么梯度指向最陡峭的上升方向？

梯度是一个向量，包含损失函数对每个参数的偏导数。从数学上可以证明，**梯度方向是函数在该点上升最快的方向**。因此，梯度的**反方向**就是下降最快的方向。

直观理解：想象你站在一座山上，闭着眼睛。你想要下山（降低损失），可以双脚感受地面，往最陡的下坡方向迈出一步。梯度就是告诉你哪个方向最陡。

#### 单参数梯度下降

假设只有参数$w$，已知$b$：

1. 随机选取初始点 $w_0$
2. 计算梯度（切线斜率）$\\frac{\\partial L}{\\partial w}\\big|_{w=w_0}$
3. 更新参数：$w_1 \\leftarrow w_0 - \\eta \\frac{\\partial L}{\\partial w}\\big|_{w=w_0}$
4. 重复直到收敛或达到最大迭代次数

其中$\\eta$是**学习率（learning rate）**，它是一个**超参数（hyperparameter）**——由人设定而非机器学习。

#### 步伐大小由什么决定？

$$\\Delta w = -\\eta \\cdot \\frac{\\partial L}{\\partial w}$$

- **斜率大小**：斜率越大，步伐越大（陡坡大步走）
- **学习率$\\eta$**：学习率越大，步伐越大
  - $\\eta$太大：可能跳过最优点，在最小值附近震荡
  - $\\eta$太小：收敛太慢，可能永远到不了

#### 多参数梯度下降

对于两个参数$(w, b)$：

$$w_1 \\leftarrow w_0 - \\eta \\frac{\\partial L}{\\partial w}\\big|_{w=w_0, b=b_0}$$
$$b_1 \\leftarrow b_0 - \\eta \\frac{\\partial L}{\\partial b}\\big|_{w=w_0, b=b_0}$$

在PyTorch等深度学习框架中，微分由程序自动计算（自动微分）。

> **关键问题**：梯度下降一定会找到全局最小值吗？
>
> **答案**：不一定。可能会卡在**局部最小值（local minima）**——梯度为0但不是全局最低点。但实际上，在高维空间中，局部最小值问题并不像人们想象的那么严重（第3章会详细讨论）。"""))

ch01_cells.append(("markdown", """## 3. 线性模型的深入理解

### 3.1 线性模型的局限性

线性模型 $y = b + \\sum_j w_j x_j$ 有一个根本的限制：**$x_1$和$y$之间的关系只能是直线**。

无论怎么调整$w$和$b$：
- 如果$x_1$增大，$y$只能线性增大（$w > 0$）
- 无法表示非线性关系，比如"$x_1$小于某个值时$y$增大，大于某个值时$y$减小"

这种来自模型本身的限制称为**模型偏差（model bias）**。

### 3.2 用分段线性曲线逼近任意函数

**核心思想**：任何连续曲线都可以用足够多的**分段线性曲线（piecewise linear curve）** 来逼近。

分段线性曲线 = 常数项 + 多个Hard Sigmoid函数之和

**Hard Sigmoid函数**：在某个阈值左侧为水平线，中间有一段斜坡，右侧为水平线。

#### 用Sigmoid函数逼近Hard Sigmoid

直接写Hard Sigmoid不容易，可以用**Sigmoid函数**来逼近：

$$y = c \\cdot \\frac{1}{1 + e^{-(b + wx_1)}} = c \\cdot \\sigma(b + wx_1)$$

Sigmoid是一个S型函数：
- 当$x_1 \\to +\\infty$时，$y \\to c$（收敛到高度c）
- 当$x_1 \\to -\\infty$时，$y \\to 0$（趋近于0）

通过调整参数：
- **$w$**：改变斜率（坡度）
- **$b$**：左右平移
- **$c$**：改变高度

这样用不同Sigmoid函数的组合就可以逼近任意连续函数：

$$y = b + \\sum_i c_i \\sigma(b_i + w_i x_1)$$"""))

ch01_cells.append(("markdown", """### 3.3 多特征输入与矩阵表示

当有多个特征时（如考虑前28天的观看次数），需要用矩阵来表示。

对于3个特征（$x_1, x_2, x_3$）和3个Sigmoid的情况：

$$\\begin{bmatrix} r_1 \\\\ r_2 \\\\ r_3 \\end{bmatrix} = \\begin{bmatrix} b_1 \\\\ b_2 \\\\ b_3 \\end{bmatrix} + \\begin{bmatrix} w_{11} & w_{12} & w_{13} \\\\ w_{21} & w_{22} & w_{23} \\\\ w_{31} & w_{32} & w_{33} \\end{bmatrix} \\begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\end{bmatrix}$$

简化为：$\\mathbf{r} = \\mathbf{b} + \\mathbf{W}\\mathbf{x}$

然后通过Sigmoid：$\\mathbf{a} = \\sigma(\\mathbf{r})$

最终输出：$y = b + \\mathbf{c}^T \\mathbf{a}$

其中所有未知参数（$\\mathbf{W}$的行列、$\\mathbf{b}$、$\\mathbf{c}^T$、$b$）被拼成一个长向量$\\boldsymbol{\\theta}$。

#### 批量（Batch）与回合（Epoch）

实际训练时不会用全部数据计算梯度，而是使用**小批量梯度下降**：

- 将$N$笔数据随机分成多个**批量（batch）**，每个批量含$B$笔数据
- 每次用一个批量计算损失并更新参数
- 遍历所有批量一次称为一个**回合（epoch）**

**更新次数 vs 回合数**：
- 10000笔数据，批量大小B=10
- 一个回合 = 10000/10 = 1000次更新
- 如果B=100，一个回合 = 100次更新

批量大小是一个重要的超参数。"""))

ch01_cells.append(("markdown", """### 3.4 ReLU激活函数

除了Sigmoid，另一个更常用的激活函数是**ReLU（Rectified Linear Unit，修正线性单元）**：

$$\\text{ReLU}(x) = c \\cdot \\max(0, b + wx_1)$$

- 如果 $b + wx_1 < 0$，输出为0
- 如果 $b + wx_1 > 0$，输出为 $b + wx_1$

**2个ReLU可以合成1个Hard Sigmoid**。因此，如果用ReLU代替Sigmoid，需要2倍的数量。

实验结果表明：
- 线性模型（56天）：训练损失320，测试损失460
- 10个ReLU：与线性模型差不多
- **100个ReLU**：训练损失降至280，测试损失降至430（显著提升！）
- 1000个ReLU：训练损失更低，但测试损失不变

这说明了**模型容量**和**过拟合**之间的权衡。"""))

ch01_cells.append(("markdown", """### 3.5 从神经元到深度学习

把Sigmoid或ReLU称为**神经元（neuron）**，多个神经元组成的网络称为**神经网络（neural network）**。

多层网络结构：
- 输入 → 隐藏层1（如100个ReLU） → 隐藏层2 → 隐藏层3 → 输出

$$\mathbf{a} = \\sigma(\\mathbf{W}\\mathbf{x} + \\mathbf{b})$$
$$\mathbf{a}' = \\sigma(\\mathbf{W}'\\mathbf{a} + \\mathbf{b}')$$

这就是**深度学习**——把网络"叠"深。每一排称为一个**隐藏层（hidden layer）**。

**深度学习发展史（ImageNet错误率）**：
- 2012 AlexNet（8层）：16.4%
- 2014 VGG（19层）：7.3%
- GoogleNet（22层）：6.7%
- ResNet（152层）：3.57%

实验表明：3层网络（训练损失140，测试损失380），4层网络（训练损失100但测试损失440）——出现了**过拟合（overfitting）**。

**反向传播（Backpropagation）** 是高效计算梯度的方法，是深度学习训练的基石。"""))

ch01_cells.append(("markdown", """## 4. 机器学习完整框架总结

### 训练三步骤

1. **定义模型** $f_{\\boldsymbol{\\theta}}(\\mathbf{x})$：写出带未知参数$\\boldsymbol{\\theta}$的函数
2. **定义损失** $L(\\boldsymbol{\\theta})$：评估参数好坏
3. **优化求解** $\\boldsymbol{\\theta}^* = \\arg\\min_{\\boldsymbol{\\theta}} L(\\boldsymbol{\\theta})$

### 训练数据与测试数据

$$\\text{训练数据：}\\{(\\mathbf{x}_1, y_1), (\\mathbf{x}_2, y_2), ..., (\\mathbf{x}_N, y_N)\\}$$
$$\\text{测试数据：}\\{\\mathbf{x}_{N+1}, \\mathbf{x}_{N+2}, ..., \\mathbf{x}_{N+M}\\}$$

测试数据只有输入$x$，没有标签$y$。

### 超参数 vs 参数

| | 参数（Parameter） | 超参数（Hyperparameter） |
|---|---|---|
| 谁决定 | 机器从数据中学出 | 人手动设定 |
| 举例 | $w$, $b$（权重和偏置） | 学习率$\\eta$、批量大小、Sigmoid个数、网络层数 |
| 寻找方式 | 梯度下降等优化算法 | 经验、网格搜索、交叉验证 |

> **重要提示**：调整超参数不能用测试集！应该用验证集或交叉验证，否则会过拟合到测试集上。"""))

ch01_cells.append(("code", """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 示例1：单变量线性回归 —— 梯度下降的完整实现
# 对应书中 y = b + w*x1 的例子
# ============================================================

# 生成模拟数据：y = 0.97*x + 100 + noise
torch.manual_seed(42)
N = 100
x = torch.randn(N, 1) * 100 + 500  # 模拟观看次数
true_w, true_b = 0.97, 100.0
y = true_w * x + true_b + torch.randn(N, 1) * 50

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 训练
lr = 1e-6
losses = []
for epoch in range(1000):
    # 前向传播
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()  # MSE损失

    # 反向传播
    loss.backward()

    # 梯度下降更新（手动）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    losses.append(loss.item())

print(f"学习到的参数: w={w.item():.4f} (真实值{true_w}), b={b.item():.2f} (真实值{true_b})")
print(f"最终损失: {losses[-1]:.2f}")

# 可视化损失下降
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('训练损失下降曲线')
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label='真实数据')
x_line = torch.linspace(x.min(), x.max(), 100).reshape(-1, 1)
with torch.no_grad():
    y_line = w * x_line + b
plt.plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, label='拟合直线')
plt.xlabel('x (前一天观看次数)')
plt.ylabel('y (当天观看次数)')
plt.title('线性回归拟合结果')
plt.legend()
plt.tight_layout()
plt.show()
"""))

ch01_cells.append(("code", """# ============================================================
# 示例2：多特征 + Sigmoid 的非线性模型
# 对应书中 y = b + sum_i c_i * sigmoid(b_i + sum_j w_ij * x_j)
# ============================================================

class FlexibleModel(nn.Module):
    def __init__(self, n_features, n_sigmoids):
        super().__init__()
        # W: (n_sigmoids, n_features) 对应书中的 w_ij
        self.linear = nn.Linear(n_features, n_sigmoids)
        # 输出层: c^T * a + b
        self.output = nn.Linear(n_sigmoids, 1)

    def forward(self, x):
        # x: (batch, n_features)
        a = torch.sigmoid(self.linear(x))  # sigmoid(r) where r = Wx + b
        return self.output(a)  # c^T * a + b

# 生成非线性数据
torch.manual_seed(42)
n_features = 3
X = torch.randn(200, n_features)
y_true = 2 * torch.sin(X[:, 0:1]) + 0.5 * X[:, 1:2]**2 - X[:, 2:3] + torch.randn(200, 1) * 0.3

# 训练对比：少量Sigmoid vs 多量Sigmoid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, n_sigmoids in enumerate([5, 50]):
    model = FlexibleModel(n_features, n_sigmoids)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(2000):
        y_pred = model(X)
        loss = ((y_pred - y_true)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    axes[idx].plot(losses)
    axes[idx].set_title(f'Sigmoid数量 = {n_sigmoids} | 最终损失 = {losses[-1]:.4f}')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('MSE Loss')
    axes[idx].set_yscale('log')
    print(f"Sigmoid数量={n_sigmoids}, 最终损失={losses[-1]:.4f}")

plt.suptitle('模型容量（Sigmoid数量）对训练的影响')
plt.tight_layout()
plt.show()
"""))

ch01_cells.append(("code", """# ============================================================
# 示例3：批量（Batch）和回合（Epoch）的理解
# ============================================================

from torch.utils.data import DataLoader, TensorDataset

# 创建数据集
dataset = TensorDataset(X, y_true)
N = len(dataset)

# 对比不同批量大小
batch_sizes = [1, 10, 50, 200]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, bs in enumerate(batch_sizes):
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    model = FlexibleModel(n_features, 30)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses = []
    updates_per_epoch = (N + bs - 1) // bs  # 每个epoch的更新次数

    for epoch in range(50):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            y_pred = model(batch_x)
            loss = ((y_pred - batch_y)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    ax = axes[idx // 2][idx % 2]
    ax.plot(losses)
    ax.set_title(f'批量大小 B={bs} | 每epoch更新{updates_per_epoch}次')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')

plt.suptitle('不同批量大小对训练的影响 (SGD)')
plt.tight_layout()
plt.show()

print("注意：批量大小=1（随机梯度下降）loss震荡大但逃逸能力强")
print("批量大小=N（批量梯度下降）loss平滑但可能卡在局部最小值")
"""))

ch01_cells.append(("code", """# ============================================================
# 示例4：学习率对梯度下降的影响
# ============================================================

torch.manual_seed(42)
x_simple = torch.randn(50, 1) * 3
y_simple = 2.5 * x_simple + 1.0 + torch.randn(50, 1) * 0.5

learning_rates = [0.001, 0.01, 0.05, 0.5]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, lr in enumerate(learning_rates):
    w_lr = torch.tensor([0.0], requires_grad=True)
    b_lr = torch.tensor([0.0], requires_grad=True)

    losses = []
    w_history = []

    for epoch in range(200):
        y_pred = w_lr * x_simple + b_lr
        loss = ((y_pred - y_simple)**2).mean()
        loss.backward()

        with torch.no_grad():
            w_lr -= lr * w_lr.grad
            b_lr -= lr * b_lr.grad
            w_lr.grad.zero_()
            b_lr.grad.zero_()

        losses.append(loss.item())
        w_history.append(w_lr.item())

    ax = axes[idx // 2][idx % 2]
    ax.plot(losses)
    ax.set_title(f'学习率 $\\eta$={lr} | 最终w={w_lr.item():.4f}')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('Loss')
    if lr <= 0.05:
        ax.set_yscale('log')

plt.suptitle('学习率对梯度下降的影响\\n$\\eta$过大导致震荡/发散，$\\eta$过小导致收敛过慢')
plt.tight_layout()
plt.show()
"""))

ch01_cells.append(("code", """# ============================================================
# 示例5：误差表面可视化（二维等高线图）
# ============================================================

# 固定简单问题，画出损失关于w和b的等高线
w_grid = np.linspace(0, 5, 100)
b_grid = np.linspace(-2, 4, 100)
W, B = np.meshgrid(w_grid, b_grid)

# 使用真实数据计算损失
x_np = x_simple.numpy()
y_np = y_simple.numpy()

Z = np.zeros_like(W)
for i in range(len(w_grid)):
    for j in range(len(b_grid)):
        y_pred = W[i, j] * x_np + B[i, j]
        Z[i, j] = np.mean((y_pred - y_np)**2)

plt.figure(figsize=(10, 8))
contour = plt.contourf(W, B, Z, levels=50, cmap='RdYlBu_r')
plt.colorbar(contour, label='MSE Loss')
# 标注最优点
opt_w_idx = np.unravel_index(Z.argmin(), Z.shape)
plt.plot(W[opt_w_idx], B[opt_w_idx], 'g*', markersize=20, label=f'最优解 w={W[opt_w_idx]:.2f}, b={B[opt_w_idx]:.2f}')

# 模拟梯度下降路径
w_path = [0.2]
b_path = [3.5]
w_curr, b_curr = 0.2, 3.5
lr_path = 0.01
for _ in range(50):
    grad_w = np.mean(2 * (w_curr * x_np + b_curr - y_np) * x_np)
    grad_b = np.mean(2 * (w_curr * x_np + b_curr - y_np))
    w_curr -= lr_path * grad_w
    b_curr -= lr_path * grad_b
    w_path.append(w_curr)
    b_path.append(b_curr)

plt.plot(w_path, b_path, 'k-', linewidth=2, label='梯度下降路径')
plt.plot(w_path[0], b_path[0], 'ko', markersize=8, label='初始点')
plt.xlabel('w (权重)')
plt.ylabel('b (偏置)')
plt.title('误差表面与梯度下降路径\\n蓝色=低损失（好），红色=高损失（差）')
plt.legend()
plt.show()
"""))

ch01_cells.append(("markdown", """## 5. 常见误区与注意事项

### 误区1：梯度下降一定会找到全局最优解
**真相**：梯度下降可能卡在局部最小值或鞍点。但在高维空间中，真正的局部最小值很罕见，更多遇到的是鞍点（详见第3章）。

### 误区2：层数越深效果一定越好
**真相**：更深的网络在训练集上确实有更低的损失，但在测试集上可能因为**过拟合**而表现更差。需要在模型容量和泛化能力之间找平衡。

### 误区3：损失函数必须是MAE或MSE
**真相**：损失函数可以自己定义！选择取决于任务需求。MSE对大的误差惩罚更重（平方惩罚），MAE对所有误差平等对待。

### 误区4：Sigmoid越多模型越好
**真相**：Sigmoid数量是超参数。太少→模型偏差（欠拟合），太多→过拟合。需要用验证集来选择。

## 6. 关键公式汇总

| 公式 | 含义 |
|------|------|
| $y = b + wx_1$ | 线性模型 |
| $y = b + \\sum_j w_j x_j$ | 多特征线性模型 |
| $y = c \\cdot \\sigma(b + wx_1)$ | Sigmoid激活 |
| $y = c \\cdot \\max(0, b + wx_1)$ | ReLU激活 |
| $y = b + \\sum_i c_i \\sigma(b_i + \\sum_j w_{ij} x_j)$ | 通用非线性模型 |
| $L = \\frac{1}{N}\\sum_n e_n$ | 损失函数 |
| $w_1 \\leftarrow w_0 - \\eta\\frac{\\partial L}{\\partial w}$ | 梯度下降更新 |
| $\\boldsymbol{\\theta}^* = \\arg\\min_{\\boldsymbol{\\theta}} L$ | 优化目标 |

## 7. 与其他章节的连接

- **第2章**：实践中如何诊断和解决模型偏差/过拟合/优化问题
- **第3章**：深度学习优化理论（鞍点、批量与动量、自适应学习率、分类损失）
- **第4章**：用卷积神经网络替代全连接，处理图像特有的归纳偏置
"""))

ch01_cells.append(("markdown", """## 8. 本章核心要点

1. **机器学习的本质**：让机器自动找到一个函数，将输入映射到输出
2. **训练三步骤**：定义模型（带参数）→ 定义损失（评估好坏）→ 优化求解（梯度下降）
3. **线性模型到非线性模型**：通过Sigmoid/ReLU的组合，可以用分段线性曲线逼近任意连续函数
4. **梯度下降**：沿梯度反方向更新参数，步长由学习率和梯度大小共同决定
5. **批量训练**：实际用mini-batch而非全部数据，一个epoch内的更新次数取决于批量大小
6. **过拟合**：训练集表现好但测试集差，需要增加数据、限制模型复杂度、或使用正则化
7. **深度学习**：将多个隐藏层叠起来，每层包含多个神经元（Sigmoid/ReLU），层数越深表达能力越强
"""))

make_notebook(ch01_cells, os.path.join(NB_DIR, "ch01_机器学习基础.ipynb"))
print("Chapter 1 done!")

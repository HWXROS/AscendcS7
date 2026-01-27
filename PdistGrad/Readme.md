# pdist 梯度计算数学逻辑

## 概述
`pdist`（成对距离）函数用于计算输入矩阵中所有样本对之间的p-范数距离。本文档详细解释其反向传播的数学逻辑，适用于任意p-范数（包括欧氏距离、曼哈顿距离等）。

## 正向传播

### 定义
对于输入矩阵 $X \in \mathbb{R}^{n \times d}$（n个样本，d维特征），`pdist` 计算所有无序对 $(i,j)$（其中 $i<j$）的p-范数距离：

$$
d_{ij} = \|x_i - x_j\|_p = \left( \sum_{k=1}^d |x_{ik} - x_{jk}|^p \right)^{1/p}
$$

输出是长度为 $m = \frac{n(n-1)}{2}$ 的向量。

## 反向传播

### 单个距离梯度
设 $\delta_{ij} = x_i - x_j$，单个距离 $d_{ij}$ 对 $x_i$ 的梯度为：

**通用公式（$p \geq 1$）：**

$$
\frac{\partial d_{ij}}{\partial x_i} = \frac{1}{d_{ij}^{p-1}} \cdot \left( |\delta_{ij}|^{\circ(p-1)} \circ \text{sign}(\delta_{ij}) \right)
$$

等价形式：
$$
\frac{\partial d_{ij}}{\partial x_i} = \frac{\delta_{ij} \circ |\delta_{ij}|^{\circ(p-2)}}{d_{ij}^{p-1}}
$$

其中 $\circ$ 表示逐元素运算。

### 特殊p值情况

| p值 | 名称 | 梯度公式 | 备注 |
|-----|------|----------|------|
| p=1 | 曼哈顿距离 | $\frac{\partial d_{ij}}{\partial x_i} = \text{sign}(\delta_{ij})$ | 在 $\delta_{ij}=0$ 处不可导 |
| p=2 | 欧氏距离 | $\frac{\partial d_{ij}}{\partial x_i} = \frac{\delta_{ij}}{d_{ij}}$ | 最常用情况 |
| p→∞ | 切比雪夫距离 | 仅在最大绝对值维度有梯度 | 梯度稀疏 |

### 梯度对称性
$$
\frac{\partial d_{ij}}{\partial x_i} = -\frac{\partial d_{ij}}{\partial x_j}
$$

这一对称性在高效实现中至关重要。

## 总梯度计算

### 聚合公式
设损失函数 $L$ 对每个距离 $d_{ij}$ 的导数为 $\frac{\partial L}{\partial d_{ij}}$，则对样本 $x_i$ 的总梯度为：

$$
\frac{\partial L}{\partial x_i} = \sum_{j \neq i} \frac{\partial L}{\partial d_{ij}} \cdot \frac{\partial d_{ij}}{\partial x_i}
$$

### 高效计算方法
利用对称性，避免重复计算：

$$
\frac{\partial L}{\partial x_i} = 
\sum_{j>i} \frac{\partial L}{\partial d_{ij}} \cdot \frac{\partial d_{ij}}{\partial x_i} +
\sum_{j<i} \frac{\partial L}{\partial d_{ji}} \cdot \left(-\frac{\partial d_{ji}}{\partial x_j}\right)
$$

## 数值稳定性

### 除零问题处理
当 $d_{ij}$ 接近0时，需添加小常数 $\epsilon$：

```python
safe_dist = max(d_ij, epsilon)  # epsilon 通常取 1e-8
```

### p值的影响
- **p>2**：大差值被指数放大，梯度可能爆炸
- **p<1**：非凸优化，零点附近梯度剧烈变化
- **p=1**：L1范数，产生稀疏梯度
- **p=2**：最平滑，数值稳定性最好

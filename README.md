---
layout: default
mathjax: true
---
# DySymNetPP
PaddlePaddle implementation of Dynamic Symbolic Neural Network. DySymNet is a neural network architecture that can undertake Symbolic Regression task, which was proposed by Wenqiang Li et.al.  (10.48550/arXiv.2309.13705)
## 1. 核心思想
DySymNetPP的核心思想是将符号回归问题转化为对兼顾适应性与简洁性神经网络结构的优化问题进行求解。传统的符号回归方法通常依赖于启发式搜索或遗传算法，而DySymNetPP通过构建一个动态的、可学习的符号神经网络，实现了对符号表达式的自动学习和优化。
    - **动态符号网络**：通过强化学习RL引导的控制器RNN生成符号网络的不同结构，而非直接在巨大的函数空间中搜索树结构表达式。网络中的每个节点可以表示一个基本的数学运算（如加法、乘法、正余弦、对数、指数等），并且这些节点的连接方式可以根据学习过程中的反馈动态变化。
    -**搜索空间压缩**：与传统符号回归方法相比，DySymNetPP显著减少了搜索空间的大小。通过限制网络深度和宽度，有效地压缩了可能的符号表达式空间，使得模型更容易学习到有效的表达方式。
    -**平衡精度与复杂度**：结合平滑L0.

## 2. 符号网络架构

### (1) 动态结构设计

- **层数和算符自适应**：符号网络的层数（\(L\)）、每层的算符数量（\(u, v\)）及类型（如 \(+, -, \sin, \exp\)）由控制器 RNN 动态生成。
  
- **算符库**：预定义算符库 \(\mathcal{F}\)，例如：
  \[
  \mathcal{F} = \{\text{Id}, +, -, \times, \sin, \cos, \exp, \log, \cosh, ^2\}
  \]

- **网络层结构**：
  - 输入层：原始特征 \(x_0\)
  - 隐藏层：由线性映射和非线性变换组成，包括：
    - 一元算符（如 \(\sin(z)\)）
    - 二元算符（如 \(z_1 \times z_2\)）
  - 输出层：线性映射生成最终预测值
### (2) 数学表达

- **第 \(\ell\) 层输出**：
  
  \[
  \bm{h}^{(\ell)} = \left[ f_1(z_1^{(\ell)}), \ldots, f_u(z_u^{(\ell)}), g_1(z_{u+1}^{(\ell)}, z_{u+2}^{(\ell)}), \ldots, g_v(z_{u+2v-1}^{(\ell)}, z_{u+2v}^{(\ell)}) \right]
  \]
  
  其中 \(f_i\) 为一元算符，\(g_j\) 为二元算符。

---

## 3. 训练策略

### (1) 两阶段训练

1. **初步训练**：
   - 仅使用 MSE 损失优化网络权重 \(\Theta\)：
     
     \[
     \mathcal{L} = \frac{1}{n} \sum_{i=1}^n \| \tilde{f}(\bm{x}_i) - y_i \|^2
     \]
   
   - **自适应梯度裁剪**：通过滑动窗口（窗口大小 \(w=50\)）计算梯度范数，动态调整裁剪阈值 \(\gamma\)：
     
     \[
     \gamma = \frac{c}{w} \sum_{i=1}^w \sum_{\ell=1}^L \| \Theta^{(\ell)} \|_2
     \]
2. **正则化与剪枝**：
   - 加入 \(L_{0.5}^*\) 正则化项，促进权重稀疏性：
     
     \[
     \mathcal{L} = \text{MSE} + \lambda \sum_{\ell=1}^L L_{0.5}^*(\Theta^{(\ell)})
     \]
     
     其中 \(L_{0.5}^*\) 定义为：
     
     \[
     L_{0.5}^*(w) = \begin{cases}
       |w|^{1/2} & |w| \geq a \\
       \left(-\frac{w^4}{8a^3} + \frac{3w^2}{4a} + \frac{3a}{8}\right)^{1/2} & |w| < a
     \end{cases}
     \]
     （默认 \(a=0.01\)）

   - **剪枝策略**：将绝对值小于阈值（\(\beta=0.01\)）的权重置零：
     
     \[
     w_{pruned} = \begin{cases}
       0 & |w| < \beta \\
       w & \text{otherwise}
     \end{cases}
     \]

### (2) BFGS 优化常数
- **微调过程**：使用 BFGS 算法对剪枝后的网络权重（即表达式中的常数）进行二次优化：
  
  \[
  \Theta^* = \arg\min_{\Theta} \sum_{i=1}^n \left( \tilde{f}_{pruned}(x_i; \Theta) - y_i \right)^2
  \]
 ## 4. 控制器设计（RL引导）

### (1) 控制器架构

- **RNN 控制器**：通过马尔可夫决策过程（MDP）生成符号网络结构描述。
  - **输入**：前一步生成的算符或层数信息。
  - **输出**：层数、每层算符数量及类型。

### (2) 强化学习目标

- **奖励函数**：基于优化后表达式的 MSE 计算：
  
  \[
  R(f^*) = \frac{1}{1 + \text{MSE}(f^*, y)}
  \]

- **风险寻求策略梯度**：最大化奖励分布的上分位数（\(\epsilon\)-quantile），鼓励探索高性能结构：
  
  \[
  J_{\text{risk}}(\theta_c; \epsilon) = \mathbb{E}_{f \sim p(a|\theta_c)} \left[ R(f^*) \mid R(f^*) \geq R_e \right]
  \]

- **熵正则化**：增加策略的探索性：
  
  \[
  J_{\text{entropy}} = \mathbb{E} \left[ \mathcal{H}(f \mid \theta_c) \mid R(f^*) \geq R_e \right]
  \]
  

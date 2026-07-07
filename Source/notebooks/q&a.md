# Q&A

这个文件用于记录学习过程中的重要问题和整理后的答案。

使用约定：
- 当你说“把这个加入到q&a”时，我会把当前问题和回答整理成条目追加到本文件。
- 每个条目包含：问题、简明答案、关键公式/流程、补充理解。
- 内容会尽量保持可复习、可检索，而不是简单复制聊天记录。

---

## Q1：束搜索或者贪心是怎么加入到Transformer架构里面的？

### 简明答案

贪心搜索和束搜索不属于Transformer的网络层结构，它们是接在Transformer输出层之后的**解码策略**。

Transformer本体负责根据当前输入算出下一个token的概率分布：

```text
已有输入token -> Transformer -> logits -> softmax -> 下一个token概率分布
```

贪心搜索或束搜索负责决定：

```text
拿到概率分布后，下一步选择哪个token？
```

所以它们主要出现在**推理阶段**，不会改变Transformer的参数，也不会改变训练好的网络结构。

### 贪心搜索

贪心搜索每一步都选择当前概率最大的token：

$$
y_t = \arg\max_y P(y \mid y_{<t}, x)
$$

流程是：

```text
输入 <SOS>
Transformer输出下一个词概率
选择概率最大的词
把这个词拼回decoder输入
继续预测下一个词
直到生成 <EOS>
```

优点是简单、速度快；缺点是只看当前一步最优，如果前面某一步选错，后面很难补救。

### 束搜索 Beam Search

束搜索不会只保留一个候选，而是每一步保留分数最高的$k$条候选序列。这个$k$叫做beam size。

例如$k=3$：

```text
第1步：保留3个最可能的开头
第2步：把这3条候选都扩展，再从所有新候选里保留最好的3条
第3步：继续扩展
直到生成 <EOS> 或达到最大长度
```

序列分数通常用log概率相加：

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)
$$

使用log概率是因为直接连乘概率会越来越小，容易数值下溢；log之后，概率乘法可以变成加法。

### 它接在Transformer的哪个位置？

完整推理流程可以理解为：

```text
Encoder编码源句子
Decoder输入当前已生成前缀
Transformer输出logits
softmax / log_softmax得到词表概率
贪心搜索或束搜索选择下一批token
把选出的token拼回decoder输入
重复
```

### 关键理解

Transformer负责“给概率”，解码算法负责“做选择”。

贪心搜索只保留当前最优的一条路径；束搜索保留多条候选路径，因此通常比贪心更稳，但计算量也更大。

---

## Q2：Q、K、V矩阵对于不同的X有什么区别？它们是怎么生成的？FFN、LayerNorm、残差连接分别有什么作用？

### 1. 先明确：X是什么？

在Transformer里，$X$通常表示某一层的输入隐藏状态。

如果是第一层，$X$来自：

$$
X = \text{TokenEmbedding} + \text{PositionalEncoding}
$$

如果是更深的层，$X$来自上一层Transformer Block的输出。

所以$X$不是原始文字本身，而是每个token对应的向量表示：

$$
X \in \mathbb{R}^{B \times L \times d_{model}}
$$

其中：

```text
B：batch size
L：序列长度
d_model：每个token向量的维度，比如512
```

例如一句话有3个token，$d_{model}=512$，那么每个样本的$X$大概是：

```text
X = [
  token1的512维向量,
  token2的512维向量,
  token3的512维向量
]
```

### 2. Q、K、V是怎么生成的？

Q、K、V不是人工指定的，也不是三个独立输入。它们是同一个输入$X$经过三组不同的线性投影得到的：

$$
Q = XW_Q + b_Q
$$

$$
K = XW_K + b_K
$$

$$
V = XW_V + b_V
$$

很多讲解会省略bias，写成：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

其中$W_Q,W_K,W_V$是模型参数，会在训练过程中通过反向传播学出来。

如果：

$$
X \in \mathbb{R}^{B \times L \times d_{model}}
$$

并且单头注意力中：

$$
W_Q,W_K,W_V \in \mathbb{R}^{d_{model} \times d_k}
$$

那么：

$$
Q,K,V \in \mathbb{R}^{B \times L \times d_k}
$$

在多头注意力里，通常会一次性投影到$d_{model}$，再拆成多个头：

```text
X:     (B, L, 512)
Q:     (B, L, 512)
拆头后: (B, h, L, 64)   # 如果h=8，512/8=64
```

### 3. 为什么同一个X要变成Q、K、V三份？

这是注意力机制最关键的设计。

如果只用一个$X$，模型很难同时表达三件事：

```text
1. 当前token想找什么信息？
2. 每个token能不能被别人匹配上？
3. 每个token真正要提供什么内容？
```

所以Transformer把同一个$X$投影成三个角色：

| 矩阵 | 名字 | 作用 | 直觉 |
|------|------|------|------|
| $Q$ | Query | 当前token发出的查询 | 我想找什么？ |
| $K$ | Key | 每个token提供的匹配线索 | 我有什么标签可被匹配？ |
| $V$ | Value | 每个token真正携带的内容 | 如果你关注我，我给你什么信息？ |

注意力不是直接用$Q$输出结果，而是：

```text
Q和K算相似度
相似度变成注意力权重
注意力权重再加权汇总V
```

公式是：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这里可以拆成三步理解：

第一步，算匹配分数：

$$
S = QK^T
$$

第$i$个token的$Q_i$会和第$j$个token的$K_j$做点积：

$$
S_{ij} = Q_i \cdot K_j
$$

如果$S_{ij}$大，说明第$i$个token很关注第$j$个token。

第二步，缩放并softmax：

$$
A = \text{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
$$

$\sqrt{d_k}$用于避免点积值过大。维度越高，点积的数值波动通常越大；如果不缩放，softmax可能变得过于尖锐，训练不稳定。

第三步，用注意力权重汇总Value：

$$
O = AV
$$

也就是说，最后真正被汇总的信息来自$V$，而$Q$和$K$主要负责决定“看谁、看多少”。

### 4. 对不同的X，Q、K、V有什么区别？

如果$X$不同，那么生成的$Q,K,V$也会不同。

因为$Q,K,V$都是$X$乘以可学习矩阵得到的：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

例如：

```text
句子1: I love AI
句子2: I hate AI
```

这两个句子的token embedding和上下文表示不同，所以它们的$X$不同。

于是：

$$
X^{(1)} \ne X^{(2)}
$$

通常就会有：

$$
X^{(1)}W_Q \ne X^{(2)}W_Q
$$

$$
X^{(1)}W_K \ne X^{(2)}W_K
$$

$$
X^{(1)}W_V \ne X^{(2)}W_V
$$

但注意：虽然不同句子的$Q,K,V$不同，使用的$W_Q,W_K,W_V$是同一套参数。

也就是说：

```text
不同输入 -> X不同 -> Q/K/V不同
同一个模型 -> W_Q/W_K/W_V共享
训练过程 -> 学的是W_Q/W_K/W_V这些投影方式
```

### 5. 同一个X生成的Q、K、V为什么也不同？

因为它们乘的是不同的权重矩阵：

$$
W_Q \ne W_K \ne W_V
$$

即使输入都是同一个$X$：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

得到的三个结果也通常不同。

这就像同一段文字可以被三种不同方式加工：

```text
Q投影：把它加工成“查询用表示”
K投影：把它加工成“匹配用表示”
V投影：把它加工成“内容用表示”
```

这三个投影矩阵在训练中会自动分工。

### 6. Self-Attention里的Q、K、V

在Self-Attention中，Q、K、V都来自同一个序列。

编码器自注意力：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

含义是：

```text
一句话内部的每个token，都可以去看这句话里的其他token。
```

比如句子：

```text
The animal didn't cross the street because it was tired.
```

模型需要判断`it`指代什么。Self-Attention允许`it`这个位置直接关注前面的`animal`。

所以Self-Attention的作用是：

```text
让序列内部的token互相交流，形成带上下文的表示。
```

### 7. Cross-Attention里的Q、K、V

在编码器-解码器结构中，Cross-Attention的Q、K、V来源不同。

通常：

$$
Q = YW_Q
$$

$$
K = EW_K,\quad V = EW_V
$$

其中：

```text
Y：解码器当前隐藏状态
E：编码器输出，也叫encoder memory
```

直觉是：

```text
Q来自解码器：我现在要生成目标句子的这个位置，我需要什么？
K来自编码器：源句子里每个位置有什么可匹配线索？
V来自编码器：源句子里每个位置真正提供什么内容？
```

所以Cross-Attention不是让目标句子内部互相看，而是让目标句子去读取源句子。

一句话：

> Self-Attention是“句子内部交流”，Cross-Attention是“解码器向编码器查询信息”。

### 8. FFN在Transformer里做什么？

FFN是Feed-Forward Network，前馈网络。它通常接在注意力层后面。

公式是：

$$
\text{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
$$

其中$\sigma$可以是ReLU、GELU、SwiGLU等激活函数。

常见维度变化是：

```text
d_model -> d_ff -> d_model
512     -> 2048 -> 512
```

也就是先升维，再非线性激活，再降回原维度。

FFN的作用可以这样理解：

```text
Attention：负责不同token之间的信息交流
FFN：负责每个token内部的信息加工
```

注意力层输出后，每个token已经拿到了上下文信息。FFN接着对每个token的表示做进一步变换，相当于“消化、加工、提炼”刚刚从其他token那里拿到的信息。

FFN有几个重要特点：

1. **逐位置处理**

FFN对每个token单独处理：

```text
x_1 -> 同一个FFN -> y_1
x_2 -> 同一个FFN -> y_2
x_3 -> 同一个FFN -> y_3
```

它不负责token之间的交流，交流已经由Attention完成。

2. **参数共享**

不同位置使用同一个FFN参数。这样模型可以把同一种“加工规则”应用到所有token位置。

3. **提供非线性能力**

如果没有FFN，Transformer会过于依赖注意力的加权汇总。FFN通过激活函数引入非线性，让模型表达能力更强。

一句话：

> Attention让token互相交流，FFN让每个token在交流之后继续思考。

### 9. LayerNorm在Transformer里做什么？

LayerNorm用于稳定每个token隐藏向量的数值分布。

对一个token向量：

$$
x = [x_1,x_2,\dots,x_d]
$$

LayerNorm先计算这个token内部所有维度的均值：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

再计算方差：

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

然后归一化：

$$
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

最后加上可学习的缩放和平移：

$$
y_i = \gamma \hat{x}_i + \beta
$$

LayerNorm的作用：

```text
1. 让每层输出的数值范围更稳定
2. 让深层Transformer更容易训练
3. 减少梯度爆炸或梯度消失的风险
4. 不依赖batch大小，适合NLP中的变长序列
```

为什么Transformer常用LayerNorm而不是BatchNorm？

BatchNorm依赖batch统计量，而NLP里经常有：

```text
不同句子长度不同
batch size可能较小
推理时可能一个样本一个样本生成
自回归生成时序列逐步变长
```

LayerNorm是对每个token自己的隐藏维度做归一化，不需要看batch里的其他样本，所以更适合Transformer。

### 10. 残差连接在Transformer里做什么？

残差连接的形式是：

$$
y = x + F(x)
$$

也就是模块输出不是只用$F(x)$，而是把输入$x$也加回来。

在Transformer里常见形式是：

$$
H = \text{LayerNorm}(X + \text{Attention}(X))
$$

$$
Y = \text{LayerNorm}(H + \text{FFN}(H))
$$

残差连接的作用：

1. **保留原始信息**

如果Attention或FFN暂时没有学好，至少原来的$x$还能继续传下去。

2. **让深层网络更容易训练**

梯度可以沿着加法路径更直接地往回传，不必完全穿过复杂的子层。

3. **让每层学习“增量修改”**

没有残差时，每层都像是在重写表示：

```text
y = F(x)
```

有残差时，每层更像是在原表示基础上补充一点东西：

```text
y = x + F(x)
```

这对堆很多层非常重要。

一句话：

> 残差连接让Transformer不是每层都推翻重来，而是在原有表示上逐层修正和补充。

### 11. 三者放在一个Transformer层里怎么看？

一个经典Encoder Layer可以写成：

$$
A = \text{SelfAttention}(X)
$$

$$
H = \text{LayerNorm}(X + A)
$$

$$
F = \text{FFN}(H)
$$

$$
Y = \text{LayerNorm}(H + F)
$$

对应流程：

```text
输入X
-> Self-Attention：token之间交流信息
-> 残差连接：把原来的X保留下来
-> LayerNorm：稳定数值分布
-> FFN：每个token独立加工
-> 残差连接：保留FFN前的信息
-> LayerNorm：再次稳定
-> 输出Y
```

最好的记忆方式是：

```text
Attention：交流
FFN：加工
残差连接：保留和传递
LayerNorm：稳定
```

再更形象一点：

```text
Attention像开会：每个token去听别的token说什么。
FFN像会后整理：每个token把听到的信息消化成自己的理解。
残差连接像保留原笔记：新理解不好时，旧信息还在。
LayerNorm像整理格式：让每一层的数值尺度保持稳定。
```

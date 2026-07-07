## 7.0 Transformer完整流程重讲：从一句话到下一个词

这一节先不急着背代码，而是把Transformer当成一条信息加工流水线来看。它的核心问题是：给定一串token，模型怎样让每个位置理解上下文，并最终预测目标序列中的下一个token？

以机器翻译为例：

```
源句子: I love AI
目标句子: 我 爱 AI
```

Transformer做的事情可以概括成一句话：

> 编码器把输入序列中的每个token变成“带上下文的表示”，解码器一边看已经生成的目标token，一边查询编码器的结果，逐步预测下一个token。

### 1. 输入：token id不是语义，embedding才是向量

文本不能直接进入神经网络。第一步是分词并查表：

```
I love AI         -> [5, 38, 120]
<SOS> 我 爱       -> [1, 245, 198]   # 训练时给解码器的输入，右移一位
我 爱 AI          -> [245, 198, 120] # 训练时希望模型预测的标签
```

设batch大小为$B$，源序列长度为$L_s$，目标序列长度为$L_t$，模型维度为$d_{model}$。输入id的形状是：

$$
\text{src\_ids} \in \mathbb{N}^{B \times L_s}, \quad
\text{tgt\_ids} \in \mathbb{N}^{B \times L_t}
$$

查embedding表后得到：

$$
X_{tok} = \text{Embedding}(\text{src\_ids}) \in \mathbb{R}^{B \times L_s \times d_{model}}
$$

如果$d_{model}=512$，那么每个token都被表示成一个512维向量。

### 2. 为什么需要位置编码？

自注意力本身只看“谁和谁相关”，不天然知道顺序。也就是说，如果没有位置编码，`I love AI`和`AI love I`在注意力层看来会非常接近，因为它们只是同一组token的排列。

所以输入Transformer前要加入位置信息：

$$
X = \sqrt{d_{model}} \cdot X_{tok} + PE
$$

原始Transformer使用正弦位置编码：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中$pos$表示第几个token，$i$表示向量维度索引。偶数维用sin，奇数维用cos。这样每个位置都有独特的“位置指纹”。

**这样安排的优点：**

- token embedding负责“这个词是什么”；
- position encoding负责“这个词在哪里”；
- 两者相加后，模型同时知道语义和顺序；
- 所有token仍然保持同样的$d_{model}$维度，后续层可以反复堆叠。

### 3. 编码器：让输入中的每个token看见整句话

编码器由$N$个相同的Encoder Layer堆叠而成。每一层做两件事：

```
Self-Attention:  让token之间交换信息
FFN:             对每个token收到的信息做非线性加工
```

一个编码器层的公式是：

$$
A = \text{MultiHeadAttention}(X, X, X)
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

这里的$X + A$和$H + F$是残差连接。残差连接保留原始信息，让深层网络更容易训练；LayerNorm稳定每个token向量的数值分布。

### 4. 自注意力：Q、K、V到底在算什么？

自注意力可以理解成一次“查询-匹配-取信息”的过程。

对输入$X \in \mathbb{R}^{B \times L \times d_{model}}$做三个线性变换：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中：

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}
$$

直觉上：

| 符号  | 含义                          | 类比          |
| ----- | ----------------------------- | ------------- |
| $Q$ | Query，当前位置想找什么信息   | 搜索关键词    |
| $K$ | Key，每个位置能被怎样匹配     | 文档标题/索引 |
| $V$ | Value，每个位置真正提供的信息 | 文档内容      |

注意力分数由$Q$和$K$的点积得到：

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

如果$Q_i$和$K_j$点积大，表示第$i$个token很需要关注第$j$个token。除以$\sqrt{d_k}$是为了防止维度变大后点积数值过大，导致softmax过早饱和、梯度变小。

然后做softmax得到权重：

$$
\alpha_{ij} = \frac{\exp(S_{ij})}{\sum_{j'=1}^{L}\exp(S_{ij'})}
$$

最后用权重对$V$加权求和：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

如果形状为$B=2, L=3, d_{model}=512, h=8, d_k=64$：

```
X:      (2, 3, 512)
Q,K,V:  (2, 8, 3, 64)     # 拆成8个头
QK^T:   (2, 8, 3, 3)      # 每个token对每个token的关注分数
attn:   (2, 8, 3, 3)      # softmax后的注意力权重
attn V: (2, 8, 3, 64)     # 每个头得到上下文向量
concat: (2, 3, 512)       # 拼回模型维度
```

### 5. 为什么要多头注意力？

单头注意力只有一种“看上下文”的方式。多头注意力让模型在不同子空间里并行观察关系：

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O
$$

例如一部分头可能关注主谓关系，一部分头关注代词指代，一部分头关注相邻词搭配。它们拼接后再经过$W_O$融合。

**这样安排的优点：**

- 每个头只处理$d_k=d_{model}/h$维，计算更可控；
- 不同头能学习不同关系，而不是把所有关系挤在一个注意力矩阵里；
- 拼接后仍回到$d_{model}$，方便下一层继续堆叠。

### 6. FFN：注意力负责通信，FFN负责加工

注意力层让不同token交换信息，但加权求和本身偏线性。FFN给每个位置增加非线性处理能力：

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

通常：

$$
W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}, \quad
W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}, \quad
d_{ff} \approx 4d_{model}
$$

如果$d_{model}=512$，常用$d_{ff}=2048$：

```
每个token: 512维 -> 2048维 -> ReLU/GELU -> 512维
```

注意，FFN是position-wise的：每个位置独立过同一套MLP，不直接和其他位置交互。跨位置通信已经由注意力完成了。

### 7. 解码器：既要看已生成内容，又要看源句子

解码器每层比编码器多一个Cross-Attention。它的流程是：

```
Masked Self-Attention -> Add & Norm
Cross-Attention       -> Add & Norm
FFN                   -> Add & Norm
```

对应公式：

$$
A_1 = \text{MaskedMHA}(Y, Y, Y)
$$

$$
H_1 = \text{LayerNorm}(Y + A_1)
$$

$$
A_2 = \text{MHA}(Q=H_1, K=E, V=E)
$$

$$
H_2 = \text{LayerNorm}(H_1 + A_2)
$$

$$
Y_{out} = \text{LayerNorm}(H_2 + \text{FFN}(H_2))
$$

其中$E$是编码器最终输出。

#### 7.1 Masked Self-Attention：为什么不能看未来？

训练时，目标序列可以一次性喂进去。但预测第$t$个词时，模型只能看到$1 \sim t$位置，不能偷看$t+1$以后的答案。

所以在softmax前加因果掩码：

$$
S_{ij} =
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

softmax后，$-\infty$对应的概率就是0。

例如长度为4：

```
位置1: 可以看 [1]
位置2: 可以看 [1, 2]
位置3: 可以看 [1, 2, 3]
位置4: 可以看 [1, 2, 3, 4]
```

**这样安排的优点：**训练时所有位置可以并行计算，但每个位置仍然满足自回归因果性。

#### 7.2 Cross-Attention：解码器如何读取输入句子？

Cross-Attention中：

$$
Q = H_1 W_Q, \quad K = E W_K, \quad V = E W_V
$$

也就是：

```
Q来自解码器：我现在要生成目标句子的这个位置，我需要什么？
K来自编码器：源句子每个位置有什么可匹配的线索？
V来自编码器：源句子每个位置真正提供什么内容？
```

这一步是编码器和解码器之间的桥梁。没有Cross-Attention，解码器就只能像普通语言模型一样根据目标前缀继续写，无法稳定地根据源句子翻译。

### 8. 输出层：从隐藏向量变成词表概率

解码器最后输出：

$$
D \in \mathbb{R}^{B \times L_t \times d_{model}}
$$

通过线性层映射到目标词表大小$V_{tgt}$：

$$
\text{logits} = DW_{out} + b_{out}
$$

$$
\text{logits} \in \mathbb{R}^{B \times L_t \times V_{tgt}}
$$

对每个位置做softmax得到词表概率：

$$
P(y_t = v \mid y_{<t}, x) = \frac{\exp(z_{t,v})}{\sum_{u=1}^{V_{tgt}}\exp(z_{t,u})}
$$

训练时使用交叉熵损失：

$$
\mathcal{L} = -\sum_{t=1}^{L_t} \log P(y_t^{true} \mid y_{<t}, x)
$$

### 9. 训练和推理为什么不一样？

训练时使用Teacher Forcing：真实目标序列右移一位作为解码器输入。

```
decoder input:  <SOS> 我 爱
label:          我    爱 AI
```

这样所有位置可以并行预测，所以训练很快。

推理时没有真实答案，只能自回归生成：

```
第1步: 输入 <SOS>          -> 预测 我
第2步: 输入 <SOS> 我       -> 预测 爱
第3步: 输入 <SOS> 我 爱    -> 预测 AI
```

所以Transformer的训练高度并行，但自回归推理仍然是逐token串行的。

### 10. 为什么Transformer这样设计？

| 设计                | 解决的问题                  | 优点                                     |
| ------------------- | --------------------------- | ---------------------------------------- |
| Self-Attention      | token之间如何交换信息       | 任意两个位置一层即可直接交互，长程依赖强 |
| Multi-Head          | 一种注意力不够表达多种关系  | 不同头学习不同语义/句法/位置关系         |
| Positional Encoding | 注意力不知道顺序            | 给token注入位置信息，同时保持并行计算    |
| Residual Connection | 深层网络训练困难            | 保留原信息，缓解梯度消失，方便堆很多层   |
| LayerNorm           | 序列长度变化、小batch不稳定 | 每个token独立归一化，训练和推理一致      |
| FFN                 | 仅靠注意力表达能力不足      | 对每个位置做非线性特征加工，提升容量     |
| Mask                | 解码时不能偷看未来          | 保持因果性，同时训练时仍可并行           |
| Cross-Attention     | 输出如何利用输入            | 解码器按需查询编码器表示                 |

### 11. 一句话总流程

```
源token ids
-> embedding + position
-> N层Encoder: Self-Attention + FFN
-> encoder memory

目标前缀token ids
-> embedding + position
-> N层Decoder: Masked Self-Attention + Cross-Attention + FFN
-> vocab logits
-> softmax
-> 下一个token概率
```

最重要的理解是：

> 注意力层负责“让token之间交流”，FFN负责“让每个token独立思考”，残差和LayerNorm负责“让很多层稳定堆起来”，Mask和Cross-Attention负责“让生成过程既守因果，又能读取输入”。

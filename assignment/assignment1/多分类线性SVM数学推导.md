# 多分类线性SVM数学推导

首先，任务的定位是多分类（有多个类，预测属于哪一个类），采用支持向量机（SVM）来做。这篇文章只推导如何求导，不推导SVM的基本型及等价形式，想了解的话可以看看[这篇文章](https://zhuanlan.zhihu.com/p/31652569)，写得非常仔细。

**线性SVM**（没有用核函数）的基本型其实就等价于**基于hinge loss优化一个线性分类器**，也即 $\mathrm{s} = \mathrm{W}^T\mathrm{x}$。预测时就用线性模型计算出样本属于各个类的分数，把分数最高的类别预测为样本的类别。

约定符号表示，令 样本总数/batch大小 为 $N$，每个样本表示为一个 $D$ 维向量，样本共分为 $C$ 个不同类别。于是有 $\mathrm{s} \in \mathbb{R}^{C}$，$\mathrm{W}=[\mathrm{w}^{(1)}, \mathrm{w}^{(2)}, ..., \mathrm{w}^{(C)}] \in \mathbb{R}^{D \times C}$ 和 $\mathrm{X}=[\mathrm{x}^{(1)}, \mathrm{x}^{(2)}, ..., \mathrm{x}^{(N)}]^T \in \mathbb{R}^{N \times D}$，每个单独的样本 $\mathrm{x}^{(i)} \in \mathbb{R}^{D}$，$\mathrm{y}=[y^{(1)}, y^{(2)}, ..., y^{(N)}] \in \mathbb{R}^N$，每个单独的 $y^{(i)} \in \mathbb{R}$ 表示样本 $i$ 的真实类别。

单个样本的hinge loss是：

$$\ell^{(i)} = \sum_{j \neq y^{(i)}} \max (0,\ s_j^{(i)}-s_{y^{(i)}}^{(i)}+1)$$

其中 $s_j^{(i)} = {\mathrm{w}^{(j)}}^Tx^{(i)}$ 为模型为样本 $i$ 属于类别 $j$ 预测的分数。

因此，模型在整个数据集上的hinge loss就等于：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \sum_{j \neq y^{(i)}} \max (0,\ s_j^{(i)}-s_{y^{(i)}}^{(i)}+1)$$

要对这个损失函数求导，跟平常我们求导的情形似乎不太一样，因为初看之下不是很直观，看不出来损失函数与参数矩阵 $\mathrm{W}$ 的关系。实际上，我们可以不去看整个参数矩阵 $\mathrm{W}$，而是把 $\mathrm{W}$ 看作是 $C$ 个列向量 $\mathrm{w}^{(j)}$，这样就能和损失函数建立联系了。先分析单个样本的 hinge loss 对 $\mathrm{W}$ 求导：

$$\ell^{(i)} = \sum_{j \neq y^{(i)}} \max (0,\ s_j^{(i)}-s_{y^{(i)}}^{(i)}+1) = \sum_{j \neq y^{(i)}} \max (0,\ {\mathrm{w}^{(j)}}^Tx^{(i)}-{\mathrm{w}^{y^{(i)}}}^Tx^{(i)}+1)$$

可以看到参与计算的只有 $\mathrm{w}^{(j)}$ 和 $\mathrm{w}^{y^{(i)}}$，而非整个参数矩阵 $\mathrm{W}$，所以我们只需要更新  $\mathrm{W}$中的对应 $j$ 和 $y^{(i)}$ 这两列的参数，计算梯度时也是一样。$\ell^{(i)}$ 是一个求和，我们只需要求各个 $j$ 对应的 $g(\mathrm{W}) = \max (0,\ s_j^{(i)}-s_{y^{(i)}}^{(i)}+1)$ 关于 $\mathrm{W}$ 的导数，然后求和就能得到 $\ell_i$ 关于 $\mathrm{W}$ 的导数了。

当 $s_j^{(i)}-s_{y^{(i)}}^{(i)}+1 \le 0$ 时，$g(\mathrm{W})$ 就等于0，与参数无关，所以此时无需计算梯度。只有 $s_j^{(i)}-s_{y^{(i)}}^{(i)}+1 \gt 0$ 才需要计算，此时可以令 $g(\mathrm{W})$ 分别对 $\mathrm{w}^{(j)}$、$\mathrm{w}^{y^{(i)}}$ 求导：

$$\frac{\partial g(\mathrm{W})} {\partial \mathrm{w}^{(j)}} = \frac{\partial ({\mathrm{w}^{(j)}}^Tx_i-{\mathrm{w}^{y^{(i)}}}^Tx_i+1)} {\partial \mathrm{w}^{(j)}} = x^{(i)}$$

$$\frac{\partial g(\mathrm{W})} {\partial \mathrm{w}^{y^{(i)}}} = \frac{\partial ({\mathrm{w}^{(j)}}^Tx_i-{\mathrm{w}^{y^{(i)}}}^Tx_i+1)} {\partial \mathrm{w}^{y^{(i)}}} = -x^{(i)}$$

至此就成功求出来了，后续只要进行求和求平均等操作就可以求出 $\frac{\partial \mathcal{L}}{\partial \mathrm{W}}$ 了。

经过这个例子，可以得到一个经验：**求导的关键就是建立函数与变量之间的联系**，另外，**求导不必执着于整体**，观察对某个/些分量的求导结果，**一次只研究一个元素**，然后再汇总，往往有惊喜。
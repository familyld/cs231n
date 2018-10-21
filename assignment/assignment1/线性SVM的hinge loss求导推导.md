# 线性SVM的Hinge loss求导推导

首先，任务的定位是多分类，采用线性模型，也即 $\mathrm{y} = \mathrm{W}^T\mathrm{x}$。

约定符号表示，样本总数为 $N$，每个样本表示为一个 $D$ 维向量，样本共分为 $C$ 个不同类别。于是有 $\mathrm{y} \in \mathbb{R}^{C}$，$\mathrm{W} \in \mathbb{R}^{D \times C}$ 和 $\mathrm{X} \in \mathbb{R}^{N \times D}$，每个单独的样本 $x_i \in \mathbb{R}^{D}$。

单个样本的hinge loss是：

$$\ell_i = \sum_{j \neq y_i} \max (0,\ s_j-s_{y_i}+1)$$

其中 $s_j =\mathrm{W}[:,\  j]^T\mathrm{X}[i,\ :] = {\mathrm{w}^j}^Tx_i$ 为模型为样本 $i$ 属于类别 $j$ 预测的分数，$y_i$ 为 样本的真实类别。

因此，模型在整个数据集上的hinge loss就等于：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \sum_{j \neq y_i} \max (0,\ s_j-s_{y_i}+1)$$

要对这个损失函数求导，跟平常我们求导的情形似乎不太一样，因为初看之下不是很直观，看不出来损失函数与参数矩阵 $\mathrm{W}$ 的关系。实际上，我们可以不去看整个参数矩阵 $\mathrm{W}$，而是把 $\mathrm{W}$ 看作是 $C$ 个列向量 $\mathrm{w}^j$，这样就能和损失函数建立联系了。先分析单个样本的 hinge loss 对 $\mathrm{W}$ 求导：

$$\ell_i = \sum_{j \neq y_i} \max (0,\ s_j-s_{y_i}+1) = \sum_{j \neq y_i} \max (0,\ {\mathrm{w}^j}^Tx_i-{\mathrm{w}^{y_i}}^Tx_{y_i}i+1)$$

可以看到参与计算的只有 $\mathrm{w}^j$ 和 $\mathrm{w}^{y_i}$，而非整个参数矩阵 $\mathrm{W}$，所以我们只需要更新  $\mathrm{W}$中的对应 $j$ 和 $y_i$ 这两列的参数，计算梯度时也是一样。$\ell_i$ 是一个求和，我们只需要求各个 $j$ 对应的 $g(\mathrm{W}) = \max (0,\ s_j-s_{y_i}+1)$ 关于 $\mathrm{W}$ 的导数在求和就能得到 $\ell_i$ 关于 $\mathrm{W}$ 的导数了。

当 $s_j-s_{y_i}+1 \le 0$ 时，$g(\mathrm{W})$ 就等于0，与参数无关，所以此时无需计算梯度。只有 $s_j-s_{y_i}+1 \gt 0$ 才需要计算，此时可以令 $g(\mathrm{W})$ 分别对 $\mathrm{w}^j$、$\mathrm{w}^{y_i}$ 求导：

$$\frac{\partial g(\mathrm{W})} {\partial \mathrm{w}^j} = \frac{\partial ({\mathrm{w}^j}^Tx_i-{\mathrm{w}^{y_i}}^Tx_{y_i}i+1)} {\partial \mathrm{w}^j} = x_i$$

$$\frac{\partial g(\mathrm{W})} {\partial \mathrm{w}^{y_i}} = \frac{\partial ({\mathrm{w}^j}^Tx_i-{\mathrm{w}^{y_i}}^Tx_{y_i}i+1)} {\partial \mathrm{w}^{y_i}} = -x_i$$

至此就成功求出来了，后续只要进行求和求平均等操作就可以了，这些操作都与变量 $\mathrm{W}$ 无关。

经过这个例子，可以得到一个经验：**对特殊函数进行向量/矩阵求导，关键就是建立函数与变量之间的联系**，另外，**要建立向量化思想**，求导未必是整体求导，可能只是对某些分量进行求导，只会更新某些分量而非整个参数矩阵。建立起联系后，再借助一点**矩阵求导**的知识就能得到答案了。
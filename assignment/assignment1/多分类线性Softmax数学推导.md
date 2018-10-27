# 多分类线性Softmax数学推导

## Softmax本体
首先，**Softmax是一个激活函数，既不是模型也不是损失函数**！它跟SVM也不是同一个层次的概念，SVM包含一整套理论体系，有时用来特指模型。这里说的 “线性Softmax” 其实指的是使用Softmax函数对线性分类器的输出进行处理（即 $\mathrm{a} = \text{Softmax}(\mathrm{W}^T\mathrm{x})$， 合起来也可以叫做**Softmax层**，常用作**多分类神经网络的最后一层**）。

任务的定位同样是多分类（有多个类，预测属于哪一个类），Softmax 除了可以做多分类还可以做**多标签分类**（一个样本属于多个类）。经过 Softmax 函数处理后，一个样本对应各类的分数就变成了属于各类的概率（和为1），普通分类时就预测为概率最大的类，多标签分类时就预测为概率最大的前几个类。

约定符号表示，和 [多分类线性SVM数学推导](https://github.com/familyld/cs231n/blob/master/assignment/assignment1/%E5%A4%9A%E5%88%86%E7%B1%BB%E7%BA%BF%E6%80%A7SVM%E6%95%B0%E5%AD%A6%E6%8E%A8%E5%AF%BC.md) 一样，令 样本总数/batch大小 为 $N$，每个样本表示为一个 $D$ 维向量，样本共分为 $C$ 个不同类别。于是有 $\mathrm{W}=[\mathrm{w}^{(1)}, \mathrm{w}^{(2)}, \cdots, \mathrm{w}^{(C)}] \in \mathbb{R}^{D \times C}$ 和 $\mathrm{X}=[\mathrm{x}^{(1)}, \mathrm{x}^{(2)}, \cdots, \mathrm{x}^{(N)}]^T \in \mathbb{R}^{N \times D}$，每个单独的样本 $\mathrm{x}^{(i)} \in \mathbb{R}^{D}$，$\mathrm{y}=[y^{(1)}, y^{(2)}, \cdots, y^{(N)}] \in \mathbb{R}^N$，每个单独的 $y^{(i)} \in \mathbb{R}$ 表示样本 $i$ 的真实类别。

用 $\mathrm{s} = \mathrm{W}^T\mathrm{x}$ 计算出各类别的预测分数，用 Softmax 处理后输出 $\mathrm{a} = \text{Softmax}(\mathrm{s})$，也即属于各类别的概率。 $\mathrm{s} \in \mathbb{R}^{C}$ 和 $\mathrm{a} \in \mathbb{R}^{C}$。

先看看 Softmax 函数的定义：

$$\begin{split}
\text{Softmax}(\mathrm{s}^{(i)}) &= \mathrm{a}^{(i)} = [a^{(i)}_1, a^{(i)}_2, \cdots, a^{(i)}_C]^T\\
\text{其中，} a^{(i)}_j &= \frac{e^{s^{(i)}_j}}{\sum_{k=1}^C e^{s^{(i)}_k}}
\end{split}
$$

其中 $s^{(i)}_j$ 表示第 $i$ 个样本第 $j$ 类的预测分数。可以看出，Softmax 其实就是用指数函数处理了一下各类的预测分数，然后再归一化。为什么要用指数函数呢？因为要**模仿 max 函数的行为**，max 函数只取最大的，Softmax 则是比较 soft 的一种处理方式，让大的更大（如果犯错就会惩罚得更严重），小的更小。**而且 max 函数有间断点，指数函数却是在整个定义域上都是可导的**！注意，由于指数函数在 自变量>0 的部分增长得很快，为了避免发生**数值溢出**，每个类的预测分数都要减去预测分数最大值。因为指数函数在定义域上单调递增，所以减去最大值并不影响分数的排序：

$$\begin{split}
s_{\max}^{(i)} &= \max(\mathrm{s}^{(i)})\\
a^{(i)}_j &= \frac{e^{s^{(i)}_j - s_{\max}^{(i)}}}{\sum_{k=1}^C e^{s^{(i)}_k - s_{\max}^{(i)}}}
\end{split}
$$

减去最大值后，指数函数的幂的取值范围变为 $(-\infty, 0]$，指数函数取值范围就变为 $(0, 1]$，也就不会有数值溢出的问题了。

## 损失函数
SVM用的是 hinge loss，线性Softmax用的则是交叉熵损失：

$$\ell^{(i)} = - \log a^{(i)}_{y^{(i)}}$$

注意理解！这里和一般看到的**二分类交叉熵损失** $\ell^{(i)} = - y^{(i)}\log y^{(i)}$ 有点区别，这是因为我们这里的 $y^{(i)}$ 指的是样本 $i$ 的类别，不是二元的（是=1 or 否=0）。我们使用的是**多分类交叉熵损失**：

$$\ell^{(i)} = -\sum_{k=1}^C 1_{k=y^{(i)}} a^{(i)}_{k}$$

其中 $1_{k=y^{(i)}}$ 是一个指示函数，仅当 $k=y^{(i)}$ 时取值为1，其他时候取值为0。又因为0乘任意数都为0，所以可以化为上面只有一项的形式。如果是做多标签分类学习，就不能用这个指示函数了~

得到单各样本的损失之后，我们可以计算模型在 整个训练集/batch 上的交叉熵损失：

$$\begin{split}
\mathcal{L} &= \frac{1}{N} \sum_{i=1}^N \ell^{(i)}\\ 
&= - \frac{1}{N} \sum_{i=1}^N \log a^{(i)}_{y^{(i)}}
\end{split}
$$

## 求梯度
还是先研究单样本的情况，也即求 $\frac{\partial \ell^{(i)}}{\partial W}$，按照前面单个样本的交叉熵损失公式，我们可以先建立损失函数 $\ell^{(i)}$ 与求导变量 $\mathrm{W}$ 之间的关系：

$$\begin{split}
\ell^{(i)} &= - \log a^{(i)}_{y^{(i)}}\\
&= -s_{y^{(i)}}^{(i)} + \log \sum_{k=1}^C e^{s_k^{(i)}}\\
&= -{\mathrm{w}^{(y^{(i)})}}^T\mathrm{x}^{(i)} + \log \sum_{k=1}^C e^{{\mathrm{w}^{(k)}}^T\mathrm{x}^{(i)}}
\end{split}
$$

假设我们要更新参数矩阵 $\mathrm{W}$ 的第 $j$ 列，也即 $\mathrm{w}^{(j)}$， 可以看到这里可以分为两种情况讨论（$j$ 是否等于 $y^{(i)}$）。先讨论 $j=y^{(i)}$ 的情况，此时：

$$\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}} = \frac{\partial \ell^{(i)}}{\partial a^{(i)}_{y^{(i)}}}\frac{\partial a^{(i)}_{y^{(i)}}}{\partial s_{y^{(i)}}^{(i)}} \frac{\partial  s_{y^{(i)}}^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}}$$

先求第一部分：

$$\begin{split}
\frac{\partial \ell^{(i)}}{\partial a^{(i)}_{y^{(i)}}} &= -\frac{\partial \log a^{(i)}_{y^{(i)}}}{\partial a^{(i)}_{y^{(i)}}}\\
&= - \frac{1}{a^{(i)}_{y^{(i)}}}
\end{split}
$$

再求第二部分：

$$\begin{split}
\frac{\partial a^{(i)}_{y^{(i)}}}{\partial s_{y^{(i)}}^{(i)}} &= \frac{\partial \frac{e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}}}{\partial s_{y^{(i)}}^{(i)}} \quad \text{应用除法求导法则}\\
&= \frac{e^{s^{(i)}_{y^{(i)}}}\sum_{k=1}^C e^{s^{(i)}_k}-e^{s^{(i)}_{y^{(i)}}}e^{s^{(i)}_{y^{(i)}}}}{(\sum_{k=1}^C e^{s^{(i)}_k})^2}\\
&= \frac{e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}}\frac{\sum_{k=1}^C e^{s^{(i)}_k} - e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}}\\
& = \frac{e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}}(1-\frac{e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}})\\
& = a^{(i)}_{y^{(i)}}(1-a^{(i)}_{y^{(i)}})
\end{split}
$$

最后：

$$\begin{split}
\frac{\partial s_{y^{(i)}}^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}} &= \frac{\partial {\mathrm{w}^{(y^{(i)})}}^T\mathrm{x}^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}}\\
& = \mathrm{x}^{(i)}
\end{split}
$$

所以，用链式法则汇总此来，就有：

$$\begin{split}
\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}} &=- \frac{1}{a^{(i)}_{y^{(i)}}} a^{(i)}_{y^{(i)}}(1-a^{(i)}_{y^{(i)}}) \mathrm{x}^{(i)}\\
&= (a^{(i)}_{y^{(i)}}-1) \mathrm{x}^{(i)}
\end{split}
$$

再讨论 $j \neq y^{(i)}$ 的情况，此时：

$$\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(j)}} = \frac{\partial \ell^{(i)}}{\partial a^{(i)}_{y^{(i)}}}\frac{\partial a^{(i)}_{y^{(i)}}}{\partial s_{j}^{(i)}} \frac{\partial  s_{j}^{(i)}}{\partial \mathrm{w}^{(j)}}$$

第一部分和最后一个部分和第一种情况一致，主要看第二部分：

$$\begin{split}
\frac{\partial a^{(i)}_{y^{(i)}}}{\partial s_{j}^{(i)}} &= \frac{\partial \frac{e^{s^{(i)}_{y^{(i)}}}}{\sum_{k=1}^C e^{s^{(i)}_k}}}{\partial s_{j}^{(i)}} \quad \text{分子看作常数，倒数求导}\\
&= -e^{s_{y^{(i)}}^{(i)}} \frac{e^{s_{k}^{(i)}}}{(\sum_{k=1}^C e^{s^{(i)}_k})^2}\\
&=  - \frac{e^{s_{y^{(i)}}^{(i)}}}{\sum_{k=1}^C e^{s^{(i)}_k}}\frac{e^{s_{k}^{(i)}}}{\sum_{k=1}^C e^{s^{(i)}_k}}\\
&= - a^{(i)}_{y^{(i)}}a^{(i)}_{j}
\end{split}
$$

所以，用链式法则汇总此来，就有：

$$\begin{split}
\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(j)}} &= \frac{1}{a^{(i)}_{y^{(i)}}} a^{(i)}_{y^{(i)}}a^{(i)}_{j} \mathrm{x}^{(i)}\\
&= a^{(i)}_{j} \mathrm{x}^{(i)}
\end{split}
$$

至此，就把单个样本的更新梯度全部推导出来了，总结一下就是：

$$\begin{split}
\text{参数矩阵中对应样本 } i \text{ 真实类别那一列：}\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(y^{(i)})}} &= (a^{(i)}_{y^{(i)}}-1) \mathrm{x}^{(i)}\\
\text{其他列：}\frac{\partial \ell^{(i)}}{\partial \mathrm{w}^{(j)}} &= a^{(i)}_{j} \mathrm{x}^{(i)}
\end{split}
$$

不要忘了还要对所有样本算得的梯度求平均（求和然后除以总数）。
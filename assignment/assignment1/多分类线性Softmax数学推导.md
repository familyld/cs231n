# 多分类线性Softmax数学推导

首先，**Softmax是一个激活函数，既不是模型也不是损失函数**！它跟SVM也不是同一个层次的概念，SVM包含一整套理论体系，有时用来特指模型。这里说的 “线性Softmax” 其实指的是使用Softmax函数对线性分类器的输出进行处理（即 $\mathrm{o} = \text{Softmax}(\mathrm{W}^T\mathrm{x})$， 合起来也可以叫做**Softmax层**，常用作多分类神经网络的最后一层）。

任务的定位同样是多分类（有多个类，预测属于哪一个类），Softmax除了可以做多分类还可以做**多标签分类**（一个样本属于多个类）。经过Softmax函数处理后，一个样本对应各类的分数就变成了属于各类的概率（和为1），普通分类时就预测为概率最大的类，多标签分类时就预测为概率最大的前几个类。

约定符号表示，令 样本总数/batch大小 为 $N$，每个样本表示为一个 $D$ 维向量，样本共分为 $C$ 个不同类别。于是有 $\mathrm{W} \in \mathbb{R}^{D \times C}$ 和 $\mathrm{X} \in \mathbb{R}^{N \times D}$，每个单独的样本 $x_i \in \mathbb{R}^{D}$，$\mathrm{y} \in \mathbb{R}^N$，每个单独的 $y_i$ 表示样本 $i$ 的真实类别。和 [多分类线性SVM数学推导](https://github.com/familyld/cs231n/blob/master/assignment/assignment1/%E5%A4%9A%E5%88%86%E7%B1%BB%E7%BA%BF%E6%80%A7SVM%E6%95%B0%E5%AD%A6%E6%8E%A8%E5%AF%BC.md) 一样，用 $\mathrm{s} = \mathrm{W}^T\mathrm{x}$ 表示每个类别的预测分数，用Softmax处理后输出 $\mathrm{o} = \text{Softmax}(\mathrm{s})$，因此有 $\mathrm{s} \in \mathbb{R}^{C}$ 和 $\mathrm{o} \in \mathbb{R}^{C}$。
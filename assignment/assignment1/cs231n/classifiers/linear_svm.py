import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train): # 逐个训练样本逐个类别来计算loss
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]: # 公式中求和项是跳过真实类别的
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # 差距比阈值小就要算loss，否则不用
        loss += margin
        dW[:, j] += X[i] # 只有margin大于0才需要计算梯度，其他情况下为0不用算
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
   # 小心别忘了要平均，看公式，模型在整个数据集上的hinge loss是要除以N的，上面两个循环完成了求和，这里要再除以N求平均
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W # 正则化项对W的求导
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]  # 样本总数N
  scores = np.dot(X, W)  # 先计算出所有得分
  # 得到每个样本真实类别对应的得分，注意第一个轴为什么要用np.arange(num_train)而不是用 : 来取所有行
  # 这是因为如果用 :，那么切片得到的N*N的矩阵，因为是先对第一个轴切片再对第二个轴切片的
  # 而用np.arange(num_train)则是在第一个轴分别取出 np.dot(X, W) 对应每一个样本的那一行，然后再在第二个轴取出真实类别那一列
  # 这样就能得到我们期望的N*1的向量了
  y_score = scores[np.arange(num_train), y].reshape((-1, 1))
  mask = (scores - y_score + 1) > 0  # 要计算loss的那些下标
  scores = (scores - y_score + 1) * mask  # 乘上mask的话max取0的部分就化0了~
  # 因为L_i求和是不求真实类别y_i对应的那一项的，但上面的代码中scores-y_score+1是算的，此时这一项=1肯定大于0，所以每个样本i的L_i都多算了1
  # 这里减去样本数就对了，然后再除以样本数求平均
  loss = (np.sum(scores) - num_train * 1) / num_train
  loss += reg * np.sum(W * W) # 正则化项~
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # 这里比较复杂，要想明白更新的到底是什么。首先mask矩阵跟scores矩阵行列数都是一样的，N*C，即每行对应一个样本，每列对应一个类别
  # mask_ij 为1表示模型为第 i 个样本第 j 类预测的分数减去真实类别的预测分数大于-1，此时就有梯度要算了，具体来说，mask_ij为1就要
  # 更新参数矩阵 W 的第 j 列和第 y_i 列，第 j 列的梯度加上一个一个X[i]，第 y_i 列则是减去一个。这里ds矩阵就是用来表示每个样本对每个类别
  # 贡献梯度的次数，ds的行列数和mask、scores都是一样的，下面三行代码就是统计每个样本对每个类别贡献梯度的次数。先全部初始化为贡献1次，
  # 然后乘上mask就把没有贡献梯度的那些元素变为0了。特别要注意ds_(i,y_i)，也即样本 i 真实类别 y_i 对应的那个元素，它的次数就等于这个样本
  # 预测分数减去真实类别预测分数大于-1的类别数-1，减一是因为把s_yi自己算上了。前面乘上-1是因为求导时得到的时-X[i]。
  ds = np.ones_like(scores)  # 初始化ds
  ds *= mask  # 有效的score梯度为1，无效的为0
  ds[np.arange(num_train), y] = -1 * (np.sum(mask, axis=1) - 1)  # 每个样本对应label的梯度计算了(有效的score次)，取负号
  # 统计好每个样本对每个类别贡献梯度的次数后，我们就要真正地计算梯度矩阵dW，dW与W都是D*C的，每行对应一个特征，每列对应一个类别，
  # dW_ij表示的就是特征 i 在类别 j 的梯度，它等于以各样本在类别 j 贡献梯度的次数对各样本的特征 i 进行加权求和所得的值，
  # 所以这里用 X.T 和 ds 求点积，X.T 的行 i 就对应各样本的特征 i，ds的列 j 就对应个样本在类别 j 贡献梯度的次数。 
  # 换个说法就是，每个样本 X[i] 对类别 j 贡献了 ds_ij 次，对第 k 个特征贡献的就是第 k 个特征的特征值 X[i][k]，那么要求某个特征 i 在类别 j 的梯度
  # 其实就是统计所有样本贡献的梯度和，也即前面说的加权和了。 
  dW = np.dot(X.T, ds) / num_train   # 最后才求个平均
  dW += 2 * reg * W  # 加上正则项的梯度
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

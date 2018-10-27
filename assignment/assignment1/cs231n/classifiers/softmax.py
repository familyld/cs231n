import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W) # scores的shape是 (C, )
    scores = scores - np.max(scores) # 减去最大分数，避免指数函数数值溢出
    loss += -1*scores[y[i]] + np.log(np.sum(np.exp(scores)))
    for j in range(num_classes):
      a = np.exp(scores[j]) / np.sum(np.exp(scores)) # a是一个数值，激活值
      if j == y[i]:
        dW[:, j] += (a-1) * X[i]
      else:
        dW[:, j] += a*X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W # 正则化项对W的求导
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W) # scores的shape是 (N, C)
  scores -= np.max(scores, axis=1, keepdims=True)  # 每个样本的预测分数向量都减去其最大分数，避免指数函数数值溢出
  a = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)  # softmax处理
  ds = np.copy(a)
  ds[np.arange(num_train), y] -= 1 # 两种情况中第一种（j=y^(i)）的梯度其实就是多减了一个x^(i)，所以这里先为这种情况-1
  dW = np.dot(X.T, ds) # 然后再点乘
  loss = a[np.arange(num_train), y]
  loss = -np.log(loss).sum()
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W # 正则化项对W的求导
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


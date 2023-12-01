# -*- coding=utf-8 -*-
# @File     : observer.py
# @Time     : 2023/8/2 12:23
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 需要将此文件进行解耦，因为这一部分应该归于模型训练阶段，而非backtest
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from scipy.special import xlogy

from sklearn.metrics import fbeta_score
from sklearn.preprocessing import Binarizer

from support import *


def focal_loss(y_true: Union[np.ndarray], y_pred: Union[np.ndarray], gamma=0, alpha:Union[float, int]=None,size_average=True):
    """
    todo: 怎么将precision和recall融合到这个loss
    OHEM算法虽然增加了错分类样本的权重，但是OHEM算法忽略了容易分类的样本。
    f1-beta 可以通过设定a的值来控制正负样本对总的loss的共享权重。a取比较小的值来降低负样本（多的那类样本）的权重。
    显然前面的公式3虽然可以控制正负样本的权重，但是没法控制容易分类和难分类样本的权重，于是就有了focal loss

    这里的γ称作focusing parameter，γ>=0。称为调制系数（modulating factor）为什么要加上这个调制系数呢？目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。
    This means that these easy examples still contribute to the total loss and affect the model’s learning. The authors want to avoid this problem by using a focal loss function that reduces the loss value for easy examples to almost zero.
    pt很大，说明是一个易分类样本，那么它的重要性应该下降

    Parameters
    ----------
    size_average
        是否将结果平均
    gamma
        focusing parameter/调制系数（modulating factor）为什么要加上这个调制系数呢？目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。gamma越大，对于minority样本重视程度更高
    alpha
        If alpha_t is greater, it means more weight is given to the positive class (usually the minority class in an imbalanced dataset).
        α-balanced variant of the focal loss. A common method for addressing class imbalance is to
         introduce a weighting factor α ∈ [0,1] for class 1 and 1−α for class −1. In practice α may be set by inverse class frequency or treated as a hyperparameter to set by cross validation.
    y_true
    y_pred
        requires proba

    Returns
    -------

    References
    ----------
    [1] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

    Examples
    --------
    >>>y_true=np.array([0,0,0,1,1])
    >>>y_pred=np.array([0.1,0,0.5,0.5,1])
    >>>gamma=2
    >>>alpha=1-y_true.mean()
    >>>size_average=True
    >>>focal_weight # 可以看到，minority样本权重高于majority，并且y_pred置信度较高的权重也比较小
    [-0.004 -0.    -0.1   -0.15  -0.   ]
    >>>loss
    [0.00042144 0.         0.06931472 0.10397208 0.        ]

    """
    if alpha is None:
        alpha = 1 - y_true.mean()
    assert alpha>=0 and alpha<=1
    # pt
    pt=np.where(y_true,y_pred,1-y_pred)
    # alpha_t
    alpha_t=np.where(y_true,alpha,1-alpha)

    focal_weight = -alpha_t*np.float_power(1-pt,np.full_like(pt,gamma))
    loss = xlogy(focal_weight,pt)
    inf_idx=loss==np.inf
    neg_inf_idx=loss==-np.inf
    finite_idx=np.isfinite(loss)
    loss =np.where(inf_idx,3*np.max(loss[finite_idx]),loss)
    loss = np.where(neg_inf_idx, 3 * np.min(loss[finite_idx]), loss)
    if size_average:
        return loss.mean()
    else:
        return loss.sum()

def fbeta_focal_score(y_true: Union[np.ndarray], y_pred: Union[np.ndarray], gamma=0, alpha:Union[float, int]=None, beta=1, zeta=1, size_average=True):
    """
    todo: 怎么将precision和recall融合到这个loss
    OHEM算法虽然增加了错分类样本的权重，但是OHEM算法忽略了容易分类的样本。
    f1-beta 可以通过设定a的值来控制正负样本对总的loss的共享权重。a取比较小的值来降低负样本（多的那类样本）的权重。
    显然前面的公式3虽然可以控制正负样本的权重，但是没法控制容易分类和难分类样本的权重，于是就有了focal loss

    这里的γ称作focusing parameter，γ>=0。称为调制系数（modulating factor）为什么要加上这个调制系数呢？目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。
    This means that these easy examples still contribute to the total loss and affect the model’s learning. The authors want to avoid this problem by using a focal loss function that reduces the loss value for easy examples to almost zero.
    pt很大，说明是一个易分类样本，那么它的重要性应该下降

    Parameters
    ----------
    zeta
        用于调配fbeta和focal loss的重要性
    beta
        greater the beta is, more important the recall is
    size_average
        是否将结果平均
    gamma
        focusing parameter/调制系数（modulating factor）为什么要加上这个调制系数呢？目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。gamma越大，对于minority样本重视程度更高
    alpha
        If alpha_t is greater, it means more weight is given to the positive class (usually the minority class in an imbalanced dataset).
        α-balanced variant of the focal loss. A common method for addressing class imbalance is to
         introduce a weighting factor α ∈ [0,1] for class 1 and 1−α for class −1. In practice α may be set by inverse class frequency or treated as a hyperparameter to set by cross validation.
    y_true
    y_pred
        requires proba

    Returns
    -------

    References
    ----------
    [1] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

    Examples
    --------
    >>>y_true=np.array([0,0,0,1,1])
    >>>y_pred=np.array([0.1,0,0.5,0.5,1])
    >>>gamma=2
    >>>alpha=1-y_true.mean()
    >>>size_average=True
    >>>focal_weight # 可以看到，minority样本权重高于majority，并且y_pred置信度较高的权重也比较小
    [-0.004 -0.    -0.1   -0.15  -0.   ]
    >>>loss
    [0.00042144 0.         0.06931472 0.10397208 0.        ]

    """
    if alpha is None:
        alpha = 1 - y_true.mean()
    assert alpha>=0 and alpha<=1
    # pt
    pt=np.where(y_true,y_pred,1-y_pred)
    # alpha_t
    alpha_t=np.where(y_true,alpha,1-alpha)

    focal_weight = -alpha_t*np.float_power(1-pt,np.full_like(pt,gamma))
    loss = xlogy(focal_weight,pt)
    inf_idx=loss==np.inf
    neg_inf_idx=loss==-np.inf
    finite_idx=np.isfinite(loss)
    loss =np.where(inf_idx,3*np.max(loss[finite_idx]),loss)
    loss = np.where(neg_inf_idx, 3 * np.min(loss[finite_idx]), loss)

    if size_average:
        loss= loss.mean()
    else:
        loss= loss.sum()
    y_pred=Binarizer(threshold=0.8).fit_transform(y_pred.reshape(-1,1)) # 这里设置阈值为0.5是为了更好地驱动模型预测更高的置信度
    fbeta=fbeta_score(y_true,y_pred,beta=beta,zero_division=0,average='binary') # bigger the fbeta score is, smaller the loss is
    loss= zeta*fbeta -loss # loss最优为0，fbeta最优为1，因此该loss最优为1
    return loss

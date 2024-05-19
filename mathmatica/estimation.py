import cupy as cp
import numpy as np
from typing import Union

from cuml.metrics import confusion_matrix as cuda_matrix
from sklearn.metrics import confusion_matrix as cpu_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

"""
TN    FP
FN    TP
precision = TP/(TP+FP)
recall = TP/(TP+FN)
"""

def get_accuracy(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    return (pred_Y == true_Y).sum() / len(pred_Y)


def get_confusion_matrix(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    if use_cuda:
        return cuda_matrix(true_Y, pred_Y)
    else:
        return cpu_matrix(true_Y, pred_Y)


def get_error_rate(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    return (pred_Y != true_Y).sum() / len(pred_Y)
    

def get_precision(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    if use_cuda:
        matrix = get_confusion_matrix(pred_Y, true_Y, use_cuda)
        return cp.array([matrix[i, i]/matrix[:, i].sum() for i in range(len(matrix))])
    else:
        return precision_score(true_Y, pred_Y, average=None)  # get x/sum_column


def get_recall(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    if use_cuda:
        matrix = get_confusion_matrix(pred_Y, true_Y, use_cuda)
        return cp.array([matrix[i, i]/matrix[i, :].sum() for i in range(len(matrix))])
    else:
        return recall_score(true_Y, pred_Y, average=None)  # get x/sum_row


def get_F1_score(pred_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array], use_cuda: bool):
    if use_cuda:
        matrix = get_confusion_matrix(pred_Y, true_Y, use_cuda)
        return cp.array([2*matrix[i, i]/(matrix[i, :].sum() + matrix[:, i].sum()) for i in range(len(matrix))])
    else:
        return f1_score(true_Y, pred_Y, average=None)  # get 2*x/(sum_row + sum_column)


def get_AUC_score(score_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array]):
    """
    ROC AUC usually fit for binary classification
    for multiple classification, it applies ovr (one vesus rest) strategy and shall return auc for every class
    the overall auc can be compute by mean value
    """ 
    return roc_auc_score(true_Y, score_Y, average=None, multi_class='ovr')

def get_AP_score(score_Y: Union[cp.array, np.array], true_Y: Union[cp.array, np.array]):
    """
    average precision usually fit for binary classification
    for multiple classification, it applies ovr (one vesus rest) strategy and shall return ap for every class
    the overall ap can be compute by mean value
    """ 
    return average_precision_score(true_Y, score_Y, average=None)


def get_logarithm_loss():
    pass


def get_MAE():
    pass



if __name__ == '__main__':
    pred_Y = [1, 1, 3, 1, 3, 1, 2, 1, 1, 2, 3, 3, 1, 4, 4]  #[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # [1, 1, 3, 1, 3, 1, 2, 1, 1, 2, 3, 3, 1, 4, 4]
    true_Y = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 4]  #[1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1] # [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 4]


    cpu_pred_Y, cpu_true_Y = np.array(pred_Y), np.array(true_Y)
    cuda_pred_Y, cuda_true_Y = cp.array(pred_Y), cp.array(true_Y)

    print(get_confusion_matrix(cpu_pred_Y, cpu_true_Y, False))
    print(get_precision(cpu_pred_Y, cpu_true_Y, False))
    print(get_recall(cpu_pred_Y, cpu_true_Y, False))
    print(get_F1_score(cpu_pred_Y, cpu_true_Y, False))
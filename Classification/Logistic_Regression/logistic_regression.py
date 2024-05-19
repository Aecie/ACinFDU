"""
Sigmoid is introduced to binary classification: sigmoid = (1 + exp(-x))^(-1)
Softmax is introduced to multi-classification: softmax(x_i) = exp(x_i)/sigma[1, C] exp(x_j)

cross-entropy is applied as loss function

[REF] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
import os
import sys
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from cuml.linear_model import LogisticRegression as CudaLogisticRegression
from sklearn.linear_model import LogisticRegression as CPULogisticRegression
from sklearn.metrics import roc_auc_score
from database.IrisSpecies.Iris_reader import IrisSpeciesReader

from mathmatica.estimation import get_accuracy, get_F1_score, get_precision, get_recall, get_AUC_score, get_AP_score

def example_Iris_analysis():
    use_cuda = True
    train_X, train_Y, test_X, test_Y = IrisSpeciesReader(test_rate=.98, use_cuda=use_cuda).load_data()
    model = CudaLogisticRegression(penalty='l2', multi_class='ovr') if use_cuda else CPULogisticRegression(penalty='l2', multi_class='ovr')
    classifier = model.fit(train_X, train_Y)
    pred_Y = classifier.predict(test_X)
    score_Y = classifier.predict_proba(test_X)

    print(pred_Y)
    print(test_Y)
    print(score_Y)
    print('ACC', get_accuracy(pred_Y, test_Y, use_cuda))
    print('Precision', get_precision(pred_Y, test_Y, use_cuda))
    print('Recall', get_recall(pred_Y, test_Y, use_cuda))
    print('F1', get_F1_score(pred_Y, test_Y, use_cuda))
    print('AUC', get_AUC_score(score_Y, test_Y, use_cuda))
    print('AP', get_AP_score(score_Y, test_Y, use_cuda))

    pass


example_Iris_analysis()
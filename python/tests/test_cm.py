import math
import numpy as np
import pytest
from baseline import ConfusionMatrix

Y_TRUE = [2, 0, 2, 2, 0, 1, 3, 1, 3, 3, 3, 3, 4]
Y_PRED = [0, 0, 2, 2, 0, 2, 3, 3, 3, 1, 3, 2, 4]
LABELS = ['0', '1', '2', '3', '4']

CLASS_PREC = [0.666667, 0.0, 0.5, 0.75, 1.0]
CLASS_RECALL = [1.0, 0.0, 0.666667, 0.6, 1.0]
CLASS_F1 = [0.8, 0.0, 0.571429, 0.666667, 1.0]
CLASS_SUPPORT = [2, 2, 3, 5, 1]
TOL = 1e-6

def make_mc_cm():
    cm = ConfusionMatrix(LABELS)
    for y_t, y_p in zip(Y_TRUE, Y_PRED):
        cm.add(y_t, y_p)
    return cm

def test_mc_support():
    cm = make_mc_cm()
    support = cm.get_support()
    np.testing.assert_allclose(support, CLASS_SUPPORT, TOL)

def test_mc_precision():
    cm = make_mc_cm()
    prec = cm.get_precision()
    np.testing.assert_allclose(prec, CLASS_PREC, TOL)
    wp = cm.get_weighted_precision()
    np.testing.assert_allclose(wp, 0.5833333, TOL)
    mp = cm.get_mean_precision()
    np.testing.assert_allclose(mp, 0.5833333, TOL)

def test_mc_recall():
    cm = make_mc_cm()
    recall = cm.get_recall()
    np.testing.assert_allclose(recall, CLASS_RECALL, TOL)
    wr = cm.get_weighted_recall()
    np.testing.assert_allclose(wr, 0.6153846, TOL)
    mr = cm.get_mean_recall()
    np.testing.assert_allclose(mr, 0.65333333, TOL)

def test_mc_f1():
    cm = make_mc_cm()
    f1 = cm.get_class_f()
    np.testing.assert_allclose(f1, CLASS_F1, TOL)
    wf1 = cm.get_weighted_f()
    np.testing.assert_allclose(wf1, 0.5882784, TOL)


def random_cm(k=10, size=100):
    gold = np.random.randint(0, k, size=(size,))
    data = np.random.randint(0, k, size=(size,))
    cm = ConfusionMatrix(range(k))
    cm.add_batch(gold, data)
    return cm


def explicit_binary_mcc(cm):
    """Implemented based on https://wikimedia.org/api/rest_v1/media/math/render/svg/33f3d62224f97cdef8bc559ee455c3f4815f5788"""
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    numer = TP * TN - FP * FN
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = numer / denom
    return mcc if not np.isnan(mcc) else 0


def explicit_mc_mcc(cm):
    """Implemented based on https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Multiclass_case"""
    numer = 0
    for k in range(len(cm)):
        for l in range(len(cm)):
            for m in range(len(cm)):
                numer += cm[k, k] * cm[l, m] - cm[k, l] * cm[m, k]

    denom1 = 0
    for k in range(len(cm)):
        k_row = 0
        for l in range(len(cm)):
            k_row += cm[k, l]
        other_rows = 0
        for k2 in range(len(cm)):
            if k2 != k:
                for l in range(len(cm)):
                    other_rows += cm[k2, l]
        denom1 += k_row * other_rows

    denom2 = 0
    for k in range(len(cm)):
        k_col = 0
        for l in range(len(cm)):
            k_col += cm[l, k]
        other_cols = 0
        for k2 in range(len(cm)):
            if k2 != k:
                for l in range(len(cm)):
                    other_cols += cm[l, k2]
        denom2 += k_col * other_cols

    mcc = numer / (math.sqrt(denom1) * math.sqrt(denom2))

    return mcc if not np.isnan(mcc) else 0


def test_binary_mcc():
    cm = random_cm(k=2)
    gold = explicit_binary_mcc(cm._cm)
    np.testing.assert_allclose(cm.get_mcc(), gold)


def test_mc_mcc():
    cm = random_cm()
    gold = explicit_mc_mcc(cm._cm)
    np.testing.assert_allclose(cm.get_r_k(), gold)

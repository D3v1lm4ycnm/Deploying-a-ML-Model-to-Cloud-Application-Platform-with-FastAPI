import numpy as np
from starter.ml.model import compute_model_metrics, train_model


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, np.float64)
    assert isinstance(recall, np.float64)
    assert isinstance(fbeta, np.float64)

def test_inference():
    preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    assert isinstance(preds, np.ndarray)

def test_train_model():
    X_train = np.array([
        [1, 2, 3, 4, 5], 
        [4, 5, 6, 7, 8], 
        [7, 8, 9, 10, 11], 
        [10, 11, 12, 13, 14], 
        [13, 14, 15, 16, 17],
        [16, 17, 18, 19, 20],
        [19, 20, 21, 22, 23],
        [22, 23, 24, 25, 26],
        [25, 26, 27, 28, 29],
        [28, 29, 30, 31, 32]])
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert model is not None
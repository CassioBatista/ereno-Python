"""GenericEvaluation equivalent — train and test a single classifier."""

import time
import numpy as np
from sklearn.base import clone

import python.config as config
from python.result import Result
import python.util as util


def _run_classifier(
    classifier_ext,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Result:
    clf = clone(classifier_ext.get_classifier())

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()

    if config.PRINT_TRAINING_TIME:
        print(f"Tempo de treinamento = {(t1 - t0) * 1e9:.0f} ns")

    VP = VN = FP = FN = 0
    n_classes = config.NUM_CLASSES
    confusion = [[0] * n_classes for _ in range(n_classes)]

    t_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_end = time.perf_counter()

    nc = util.normal_class
    for expected, predicted in zip(y_test, y_pred):
        # match/case requer bool Python; numpy.bool_ não passa em `case True:`
        correct   = bool(predicted == expected)
        is_normal = bool(predicted == nc)
        match (correct, is_normal):
            case (True,  True):  VN += 1
            case (True,  False): VP += 1
            case (False, True):  FN += 1
            case (False, False): FP += 1
        if expected < n_classes and predicted < n_classes:
            confusion[int(expected)][int(predicted)] += 1

    n = len(y_test)
    avg_time_ns = (t_end - t_start) * 1e9 / n if n else 0.0
    return Result(classifier_ext.get_classifier_name(), VP, FN, VN, FP, avg_time_ns, confusion)


def run_single_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Result:
    r = _run_classifier(config.SINGLE_CLASSIFIER_MODE, X_train, y_train, X_test, y_test)
    if config.CSV:
        print(
            f"{r.cx};{r.accuracy};{r.precision};"
            f"{r.recall};{r.f1score};"
            f"{r.VP};{r.VN};{r.FP};{r.FN};{r.avg_time}"
        )
    return r


def run_multiple_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[Result]:
    results = []
    for clf_ext in config.CLASSIFIERS_FOREACH:
        r = _run_classifier(clf_ext, X_train, y_train, X_test, y_test)
        results.append(r)
        if config.CSV:
            print(
                f"{clf_ext.get_classifier_name()};"
                f"{r.accuracy};{r.precision};{r.recall};{r.f1score};"
                f"{r.VP};{r.VN};{r.FP};{r.FN};{r.avg_time}"
            )
    return results

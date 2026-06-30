"""CrossValidation equivalent — k-fold cross-validation."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

import python.config as config
from python.result import Result
import python.evaluation as evaluation
import python.util as util


def run_single_classifier(
    X: np.ndarray, y: np.ndarray, total_folds: int, seed: int
) -> list[Result]:
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=seed)
    results: list[Result] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        r = evaluation.run_single_classifier(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        )
        results.append(r)
        if config.DEBUG_MODE:
            print(f"runSingleClassifier - F1:{r.f1score}")
            print(f"runSingleClassifier - Acc:{r.accuracy}")
        if config.SINGLE_FOLD_MODE:
            return [r]
    return results


def setup_and_run(
    total_folds: int,
    seed: int,
    classifier_ext,
    fs: list[int],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> list[Result]:
    config.SINGLE_CLASSIFIER_MODE = classifier_ext
    config.FEATURE_SELECTION = fs
    if X is None or y is None:
        X, y = util.load_and_filter_single_file(False)
    else:
        X, y = util.copy_and_filter(X, y, False)
    return run_single_classifier(X, y, total_folds, seed)


def run_with_instances(
    X: np.ndarray,
    y: np.ndarray,
    features: list[int],
    classifier_ext,
    print_confusion: bool = False,
) -> None:
    print(f"{classifier_ext.get_classifier_name()};", end="")
    results = setup_and_run(config.FOLDS, config.EVALUATION_SEED, classifier_ext, features, X, y)
    for i, r in enumerate(results):
        print(f"fold[{i}];{r.f1score};", end="")
    print("")
    if print_confusion:
        for fold_num, r in enumerate(results):
            print(f"Fold: {fold_num}")
            for ci, row in enumerate(r.confusion_matrix):
                print(f"Esperado: {ci};resultados:;" + ";".join(str(v) for v in row) + ";")


def print_folds(
    X: np.ndarray,
    y: np.ndarray,
    features: list[int] | None = None,
    print_summary: bool = True,
) -> None:
    from python.classifiers import all_custom
    all_results: list[list[Result]] = []
    for clf in all_custom:
        fs = features if features is not None else list(range(1, X.shape[1] + 1))
        all_results.append(setup_and_run(config.FOLDS, config.EVALUATION_SEED, clf, fs, X, y))

    if print_summary and not config.SINGLE_FOLD_MODE:
        for clf, folds in zip(all_custom, all_results):
            print(f"{clf.get_classifier_name()};", end="")
            for r in folds:
                print(f"{r.f1score};", end="")
            print("")

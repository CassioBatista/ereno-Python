"""Util equivalent — ARFF loading and feature filtering."""

import numpy as np
import python.config as config

# PEP 695
type Dataset = tuple[np.ndarray, np.ndarray, list[str]]

# Normal class index (index of first class value in the dataset)
normal_class: int = 0


def load_arff(filepath: str) -> Dataset:
    """Load an ARFF file, returning (X, y, class_values).

    X: float64 array (n_samples, n_features)
    y: int64 array  (n_samples,)
    class_values: ordered list of class label strings
    """
    global normal_class
    attributes: list[tuple[str, str]] = []
    class_values: list[str] = []
    data_rows: list[str] = []
    in_data = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            low = stripped.lower()
            if low.startswith("@relation"):
                continue
            if low.startswith("@attribute"):
                parts = stripped.split(None, 2)
                attr_name = parts[1]
                attr_type = parts[2] if len(parts) > 2 else "numeric"
                if "{" in attr_type:
                    vals = attr_type.strip("{}").split(",")
                    class_values = [v.strip() for v in vals]
                    attributes.append((attr_name, "nominal"))
                else:
                    attributes.append((attr_name, "numeric"))
                continue
            if low.startswith("@data"):
                in_data = True
                continue
            if in_data:
                data_rows.append(stripped)

    if not data_rows:
        raise ValueError(f"No data rows found in {filepath}")

    n_attrs = len(attributes)
    X_list: list[list[float]] = []
    y_list: list[int] = []

    for row in data_rows:
        vals = row.split(",")
        if len(vals) != n_attrs:
            continue
        try:
            x = [float(v.strip()) for v in vals[:-1]]
            cls_str = vals[-1].strip()
            y_val = class_values.index(cls_str) if class_values else int(float(cls_str))
            X_list.append(x)
            y_list.append(y_val)
        except (ValueError, IndexError):
            continue

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    normal_class = int(y[0]) if len(y) > 0 else 0
    return X, y, class_values


def filter_features(X: np.ndarray, feature_indices: list[int]) -> np.ndarray:
    """Keep only specified features (1-based → 0-based columns)."""
    if not feature_indices:
        return X
    cols = [i - 1 for i in sorted(feature_indices)]
    return X[:, cols]


def load_single_file(print_selection: bool = False) -> Dataset:
    return load_arff(config.DATASET)


def load_and_filter_single_file(print_selection: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = load_single_file(print_selection)
    X = filter_features(X, config.FEATURE_SELECTION)
    if print_selection and config.FEATURE_SELECTION:
        print(f"{config.FEATURE_SELECTION} - {X.shape[1]} attributes in fact.")
    return X, y


def copy_and_filter(
    X: np.ndarray, y: np.ndarray, print_selection: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    Xf = filter_features(X, config.FEATURE_SELECTION)
    if print_selection and config.FEATURE_SELECTION:
        print(f"{config.FEATURE_SELECTION} - {Xf.shape[1]} attributes in fact.")
    return Xf, y


def get_result_average(results):
    from python.result import Result
    return Result.average(results)

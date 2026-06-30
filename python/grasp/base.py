"""Abstract Grasp base class."""

import random
import time
import numpy as np

import python.config as config
from python.grasp.solution import GraspSolution


class Grasp:
    """Abstract GRASP base."""

    criteria_metric: str = "F1SCORE"

    def __init__(self):
        self.max_time: int               = 24 * 60 * 60 * 1000
        self.max_iterations: int         = 100_000
        self.max_number_evaluation: int  = 100_000
        self.max_no_improvement: int     = 100_000
        self.num_features: int           = 5

        self.iteration_number: int       = 0
        self.no_improvements: int        = 0
        self.current_time: float         = 0.0
        self.number_evaluation: int      = 0
        self.num_bit_flip_features: int  = 5
        self.begin_time: float           = 0.0
        self._best_global_solution: GraspSolution | None = None
        self._all_instances: tuple | None = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def setup_grasp_microservice(self, classifier_index: int) -> "Grasp":
        from python.util import load_single_file
        from python.classifiers import all_classifiers
        X, y, _ = load_single_file(True)
        self._all_instances = (X, y)
        if classifier_index < 0:
            print(f"Dataset: {config.DATASET}")
            print("Choose classifier:")
            for i, c in enumerate(all_classifiers):
                print(f"  ({i + 1}) {c.get_classifier_name()}")
            classifier_index = int(input()) - 1
        config.SINGLE_CLASSIFIER_MODE = all_classifiers[classifier_index]
        return self

    # ------------------------------------------------------------------
    # Solution management
    # ------------------------------------------------------------------
    def get_best_global_solution(self) -> GraspSolution | None:
        return self._best_global_solution

    def set_best_global_solution(self, sol: GraspSolution) -> None:
        self._best_global_solution = sol

    # ------------------------------------------------------------------
    # Construction phase
    # ------------------------------------------------------------------
    def build_custom_rcl(self, rcl: list[int]) -> list[int]:
        return list(rcl)

    def construct_solution(self, full_rcl: list[int]) -> GraspSolution:
        print(f"RCL: {full_rcl}")
        rcl_copy = list(full_rcl)
        solution = GraspSolution()
        while len(solution.get_array_features()) < self.num_features and rcl_copy:
            solution.add_feature(rcl_copy.pop(random.randrange(len(rcl_copy))))
        for f in rcl_copy:
            solution.add_feature_flip(f)
        print(f"Solução Inicial: {solution.get_feature_set()}")
        print(f"RCL Restante: {solution.get_rcl_features()}")
        return solution

    def reconstruct_solution(self, initial: GraspSolution) -> GraspSolution:
        full_rcl = initial.copy_rcl_features() + initial.copy_solution_features()
        print(f"RCL: {full_rcl}")
        solution = GraspSolution()
        while len(solution.get_array_features()) < self.num_features and full_rcl:
            solution.add_feature(full_rcl.pop(random.randrange(len(full_rcl))))
        for f in full_rcl:
            solution.add_feature_flip(f)
        print(f"Solução Reconstruída: {solution.get_feature_set()}")
        print(f"RCL Restante: {solution.get_rcl_features()}")
        return solution

    # ------------------------------------------------------------------
    # Evaluation — uses t-strings (PEP 750) for structured logging
    # ------------------------------------------------------------------
    def avaliar(self, solution: GraspSolution) -> GraspSolution:
        from python.util import copy_and_filter, get_result_average
        from python import cross_validation as cv

        solution.current_time_seconds = self.begin_time / 1000 - time.time()

        if config.DEBUG_MODE:
            print(f"Dataset: {config.DATASET}")
            clf_name = (
                config.SINGLE_CLASSIFIER_MODE.get_classifier_name()
                if config.SINGLE_CLASSIFIER_MODE else "None"
            )
            print(f"Classifier: {clf_name}")
            print(f"Folds: {config.FOLDS} | Seed: {config.GRASP_SEED}")

        config.FEATURE_SELECTION = solution.get_array_features()

        match config.CROSS_VALIDATION:
            case True:
                X, y = self._all_instances
                Xf, yf = copy_and_filter(X, y, False)
                results = cv.run_single_classifier(Xf, yf, config.FOLDS, config.GRASP_SEED)
                solution.evaluation = get_result_average(results)
            case False:
                if config.TRAIN is None or config.TEST is None:
                    print("ERROR: TRAIN and TEST must be set when CROSS_VALIDATION=False")
                else:
                    from python.evaluation import run_single_classifier
                    from python.util import filter_features
                    X_tr = filter_features(config.TRAIN[0], config.FEATURE_SELECTION)
                    X_te = filter_features(config.TEST[0],  config.FEATURE_SELECTION)
                    solution.evaluation = run_single_classifier(
                        X_tr, config.TRAIN[1], X_te, config.TEST[1]
                    )

        self.number_evaluation += 1
        f1 = solution.evaluation.f1score if solution.evaluation else 0.0
        print(f"EV;{self.number_evaluation};{f1};{solution.get_array_features()}")
        return solution

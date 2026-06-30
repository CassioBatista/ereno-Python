"""GraspSimple — GRASP with a single local search per iteration."""

import time
from python.grasp.base import Grasp
from python.grasp.solution import GraspSolution
from python.grasp.local_searches import busca_local
from python.grasp.neighborhood.bit_flip import BitFlip
from python.grasp.neighborhood.iwss import IWSS
from python.grasp.neighborhood.iwssr import IWSSr


class GraspSimple(Grasp):

    def run(
        self, rcl: list[int], method_name: str, neighborhood_type: str, dataset: str
    ) -> GraspSolution:
        print("Wellcome to GRASP!")
        self.begin_time = time.time() * 1000

        neighborhood = match_neighborhood(neighborhood_type, self)

        full_rcl = self.build_custom_rcl(rcl)
        initial  = self.avaliar(self.construct_solution(full_rcl))
        self.set_best_global_solution(initial.new_clone(False))
        initial  = busca_local(initial, neighborhood, self)
        if initial.is_better_than(self.get_best_global_solution(), self.criteria_metric):
            self.set_best_global_solution(initial.new_clone(False))

        while self._should_continue():
            self.iteration_number += 1
            print(f"######### ITERATION ({self.iteration_number}) #############")
            t0 = time.time() * 1000

            reconstructed = self.avaliar(self.reconstruct_solution(initial))
            if initial.is_better_than(self.get_best_global_solution(), self.criteria_metric):
                self.set_best_global_solution(initial.new_clone(False))
                self.no_improvements = 0

            reconstructed = busca_local(reconstructed, neighborhood, self)
            if reconstructed.is_better_than(self.get_best_global_solution(), self.criteria_metric):
                self.set_best_global_solution(reconstructed.new_clone(False))
                self.no_improvements = 0
            else:
                self.no_improvements += 1

            self.current_time = time.time() * 1000 - self.begin_time
            best = self.get_best_global_solution()
            f1   = best.evaluation.f1score if best.evaluation else "N/A"
            print(
                f"######### Fim ITERAÇÂO ({self.iteration_number}"
                f" / {self.current_time / 60_000:.1f}min)"
                f" - F1Score:{f1} - Conjunto={best.get_feature_set()}"
            )

        return self.get_best_global_solution()

    def _should_continue(self) -> bool:
        return (
            self.number_evaluation < self.max_number_evaluation
            and self.iteration_number < self.max_iterations
            and self.no_improvements  < self.max_no_improvement
            and self.current_time     < self.max_time
        )


def match_neighborhood(neighborhood_type: str, grasp):
    match neighborhood_type:
        case "BIT_FLIP": return BitFlip(grasp)
        case "IWSS":     return IWSS(grasp)
        case "IWSSR":    return IWSSr(grasp)
        case _:          raise ValueError(f"Unknown neighborhood: {neighborhood_type}")

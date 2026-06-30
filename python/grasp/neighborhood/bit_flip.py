"""BitFlip — swap one selected feature with one RCL feature."""

import random
from python.grasp.solution import GraspSolution
from python.grasp.neighborhood.structure import NeighborhoodStructure


class BitFlip(NeighborhoodStructure):
    def __init__(self, grasp):
        self.grasp = grasp
        self._rem_iterations   = 50
        self._rem_no_imp       = 20

    def _single_movement(self, reference: GraspSolution) -> GraspSolution:
        neighbor = reference.new_clone(True)
        n_sol, n_rcl = neighbor.get_num_selected_features(), neighbor.get_num_rcl_features()
        if n_sol == 0 or n_rcl == 0:
            raise ValueError(f"IncompleteFeatureSelection: sol={n_sol}, rcl={n_rcl}")
        rem = 0 if n_sol == 1 else random.randint(0, n_sol - 2)
        add = 0 if n_rcl == 1 else random.randint(0, n_rcl - 2)
        neighbor.replace_feature(rem, add)
        return self.grasp.avaliar(neighbor)

    def run(self, reference: GraspSolution) -> GraspSolution:
        print("Running BitFlip:")
        best_local = reference.new_clone(False)
        its, no_imp = self._rem_iterations, self._rem_no_imp

        while its > 0 and no_imp > 0:
            its -= 1; no_imp -= 1
            try:
                neighbor = self._single_movement(reference)
                if neighbor.is_better_than(best_local, self.grasp.criteria_metric):
                    best_local = neighbor.new_clone(False)
                    no_imp = 10
            except ValueError as e:
                print(f"Cancelando run: {e}")
                return best_local

        return best_local

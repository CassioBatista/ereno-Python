"""IWSSr — Incremental Wrapper-Based Subset Selection with Replacement."""

from python.grasp.solution import GraspSolution
from python.grasp.neighborhood.structure import NeighborhoodStructure


class IWSSr(NeighborhoodStructure):
    def __init__(self, grasp):
        self.grasp = grasp
        self.full_list: list[int] = []

    def run(self, seed: GraspSolution) -> GraspSolution:
        best_local = seed.new_clone(False)
        while best_local.get_num_selected_features() > 0:
            best_local.deselect_feature(0)
        self.full_list = best_local.copy_rcl_features()
        print("Running IWSSSr:")

        for rcl_index in range(best_local.get_num_rcl_features() - 1, -1, -1):
            print(
                f"IWSSr >>> adding {best_local.get_rcl_features()[rcl_index]}"
                f" > to set {best_local.get_feature_set()}"
                f" (Acc: {best_local.get_accuracy()}, F1: {best_local.get_f1_score()})"
            )
            before = best_local.new_clone(False)

            # Try swap: replace each selected feature with this RCL feature
            for sol_index in range(best_local.get_num_selected_features()):
                swap = before.new_clone(True)
                swap.deselect_feature(sol_index)
                swap.select_feature(rcl_index)
                swap = self.grasp.avaliar(swap)
                if swap.is_better_than(best_local, self.grasp.criteria_metric):
                    best_local = swap.new_clone(False)

            # Try add
            add = before.new_clone(True)
            add.select_feature(rcl_index)
            add = self.grasp.avaliar(add)
            if add.is_better_than(best_local, self.grasp.criteria_metric):
                best_local = add.new_clone(False)

        self._restore_rcl(best_local)
        return best_local

    def _restore_rcl(self, best_local: GraspSolution) -> None:
        sol = set(best_local.get_selected_features())
        rcl = set(best_local.get_rcl_features())
        for f in self.full_list:
            if f not in sol and f not in rcl:
                best_local.add_feature_rcl(f)

"""NeighborhoodStructure — abstract interface."""

from abc import ABC, abstractmethod
from python.grasp.solution import GraspSolution


class NeighborhoodStructure(ABC):
    @abstractmethod
    def run(self, seed: GraspSolution) -> GraspSolution: ...

import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm, BoundedModule


class CellVerificationSystem():
    def __init__(self, max_depth=10):
        super().__init__()
        self.max_depth = max_depth
        self.corners = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    def verify(
        self,
            verifier: BoundedModule,
            locations: torch.Tensor,
            magnitude: torch.Tensor,
            depth: int = 0,
            bound_method: str = 'IBP',
    ):
        bounded_cells = BoundedTensor(
            locations,
            PerturbationLpNorm(
                x_L=locations - magnitude,
                x_U=locations + magnitude,
            )
        )
        _, ub = verifier.compute_bounds(
            bounded_cells,
            bound_lower=False,
            method=bound_method,
        )
        mask = ub.squeeze() >= 0.0
        counterexamples = locations[mask]

        if torch.numel(counterexamples) > 0:
            print(
                f"Could not verify decrease at {counterexamples.shape[0]} "
                "cells. Splitting further"
            )
            if depth < self.max_depth:
                half_m = 0.5 * magnitude
                corners = self.corners.to(locations.device)
                new_cells = (counterexamples.unsqueeze(1) + half_m * corners.unsqueeze(0)).reshape(-1, locations.shape[1])
                counterexamples = self.verify(
                    verifier, new_cells, half_m, depth+1,
                    bound_method=bound_method,
                )
        return counterexamples

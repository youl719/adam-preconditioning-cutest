import torch
import pycutest
import numpy as np


class CutestOracle:
    """
    CUTEst oracle wrapper with fine-grained oracle cost tracking.

    Cost convention:
        function value  = 1 unit
        gradient        = 2 units
        hvp             = 4 units  (configurable via hvp_cost)

    The three accumulators ``func_units``, ``grad_units``, and ``hvp_units``
    are updated independently so that callers can attribute costs per
    iteration.
    """

    def __init__(self, problem_name: str, sifparams: dict = None, hvp_cost: int = 4):
        self.problem_name = problem_name
        self.problem = pycutest.import_problem(problem_name, sifParams=sifparams)
        self.hvp_cost = hvp_cost

        # Fine-grained counters
        self.func_units: int = 0
        self.grad_units: int = 0
        self.hvp_units: int = 0

    @property
    def oracle_units(self) -> int:
        """Total oracle cost across all three categories."""
        return self.func_units + self.grad_units + self.hvp_units

    def __call__(self, x_k: torch.Tensor, order: str):
        x_np = x_k.detach().cpu().numpy().astype(np.float64)

        def hvp_fn(v: torch.Tensor) -> torch.Tensor:
            self.hvp_units += self.hvp_cost
            v_np = v.detach().cpu().numpy().astype(np.float64)
            hvp_np = self.problem.hprod(p=v_np, x=x_np)
            return torch.tensor(hvp_np, dtype=x_k.dtype, device=x_k.device)

        if order == "0":
            f_val = self.problem.obj(x_np)
            self.func_units += 1
            return float(f_val)

        elif order == "01":
            f_val, g_val = self.problem.obj(x_np, gradient=True)
            self.func_units += 1
            self.grad_units += 2
            g_tensor = torch.tensor(g_val, dtype=x_k.dtype, device=x_k.device)
            return float(f_val), g_tensor

        elif order == "012":
            f_val, g_val = self.problem.obj(x_np, gradient=True)
            self.func_units += 1
            self.grad_units += 2
            g_tensor = torch.tensor(g_val, dtype=x_k.dtype, device=x_k.device)
            return float(f_val), g_tensor, hvp_fn

        elif order == "12":
            g_val = self.problem.grad(x_np)
            self.grad_units += 2
            g_tensor = torch.tensor(g_val, dtype=x_k.dtype, device=x_k.device)
            return g_tensor, hvp_fn

        else:
            raise ValueError(f"Unknown order request: {order}")

    def reset_counters(self):
        """Resets all benchmark counters for a new solver run."""
        self.func_units = 0
        self.grad_units = 0
        self.hvp_units = 0

    def get_initial_point(self) -> torch.Tensor:
        x0_np = self.problem.x0
        return torch.tensor(x0_np, dtype=torch.float64)

    def snapshot_counters(self) -> dict:
        """Return a snapshot of the current counters (for delta computation)."""
        return {
            "func_units": self.func_units,
            "grad_units": self.grad_units,
            "hvp_units": self.hvp_units,
        }
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "experiment_results.json"
PLOT_PATH = SCRIPT_DIR / "potential.png"


@dataclass(frozen=True)
class GridConfig:
    N_rho: int
    N_z: int
    dh: float


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    steps: int
    lbfgs_max_iter: int = 20
    history_size: int = 50


@dataclass
class ExperimentResult:
    group: str
    label: str
    optimizer: str
    init_mode: str
    steps: int
    runtime_s: float
    final_loss: float
    rel_error: float
    note: str = ""


def calculate_Q(f: torch.Tensor, dh: float, rho: torch.Tensor) -> float:
    return torch.sum(f * rho).item() * 2.0 * math.pi * (dh ** 2)


def calculate_laplace(phi: torch.Tensor, dh: float, idx_i: torch.Tensor) -> torch.Tensor:
    inv_idx_i = 1.0 / (2.0 * idx_i)
    laplacian_inner = (
        (1.0 + inv_idx_i) * phi[2:, 1:-1]
        + (1.0 - inv_idx_i) * phi[0:-2, 1:-1]
        + phi[1:-1, 2:]
        + phi[1:-1, 0:-2]
        - 4.0 * phi[1:-1, 1:-1]
    ) / (dh ** 2)
    laplacian_rho_0 = (
        4.0 * phi[1:2, 1:-1]
        + phi[0:1, 2:]
        + phi[0:1, 0:-2]
        - 6.0 * phi[0:1, 1:-1]
    ) / (dh ** 2)
    return torch.cat([laplacian_rho_0, laplacian_inner], dim=0)


def make_uniform_sphere_potential(r: torch.Tensor, q_total: float, radius: float = 12.0) -> torch.Tensor:
    r_safe = torch.clamp(r, min=1.0e-12)
    phi_out = q_total / (4.0 * math.pi * r_safe)
    phi_in = (q_total / (4.0 * math.pi * radius)) * 0.5 * (3.0 - (r / radius) ** 2)
    return torch.where(r <= radius, phi_in, phi_out)


class Solver:
    def __init__(
        self,
        grid: GridConfig,
        device: torch.device,
        optimizer_config: OptimizerConfig,
        init_mode: str = "sphere",
        seed: int = 1234,
    ) -> None:
        self.grid = grid
        self.N_rho = grid.N_rho
        self.N_z = grid.N_z
        self.dh = grid.dh
        self.device = device
        self.optimizer_config = optimizer_config
        self.init_mode = init_mode
        self.seed = seed

        self.rho_1d = torch.arange(0, self.N_rho, device=device, dtype=torch.float32) * self.dh
        self.z_1d = torch.arange(-self.N_z, self.N_z + 1, device=device, dtype=torch.float32) * self.dh
        self.rho_2d, self.z_2d = torch.meshgrid(self.rho_1d, self.z_1d, indexing="ij")

        eps = 1.0e-12
        self.r = torch.sqrt(self.rho_2d ** 2 + self.z_2d ** 2 + eps)
        self.Y = 0.25 * math.sqrt(5.0 / math.pi) * (3.0 * ((self.z_2d / self.r) ** 2) - 1.0)
        self.r0 = 10.0 * (1.0 + self.Y)
        self.f = (0.8 / (1.0 + torch.exp((self.r - self.r0) / 0.7))).detach()

        self.Q_total = calculate_Q(self.f, self.dh, self.rho_2d)
        self.sphere_guess = make_uniform_sphere_potential(self.r, self.Q_total).detach()
        self.phi_boundary = self.sphere_guess.clone()

        phi_inner_init = self.make_initial_inner(init_mode, seed)
        self.phi_inner = nn.Parameter(phi_inner_init.clone())
        self.optimizer = self.build_optimizer()

    def make_initial_inner(self, init_mode: str, seed: int) -> torch.Tensor:
        if init_mode == "zeros":
            return torch.zeros_like(self.sphere_guess[0:-1, 1:-1])
        if init_mode == "randn":
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            scale = float(self.sphere_guess[0:-1, 1:-1].abs().mean().item())
            return torch.randn(
                self.N_rho - 1,
                2 * self.N_z - 1,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ) * max(scale, 1.0e-3)
        if init_mode == "sphere":
            return self.sphere_guess[0:-1, 1:-1].clone()
        raise ValueError(f"unsupported init mode: {init_mode}")

    def build_optimizer(self) -> torch.optim.Optimizer:
        config = self.optimizer_config
        params = [self.phi_inner]
        if config.name == "sgd":
            return torch.optim.SGD(params, lr=config.lr)
        if config.name == "adam":
            return torch.optim.Adam(params, lr=config.lr)
        if config.name == "lbfgs":
            return torch.optim.LBFGS(
                params,
                lr=config.lr,
                max_iter=config.lbfgs_max_iter,
                history_size=config.history_size,
                line_search_fn="strong_wolfe",
            )
        raise ValueError(f"unsupported optimizer: {config.name}")

    def build_phi(self) -> torch.Tensor:
        phi = self.phi_boundary.clone()
        phi[0:-1, 1:-1] = self.phi_inner
        return phi

    def compute_loss(self) -> torch.Tensor:
        phi = self.build_phi()
        idx_i = 1 + torch.arange(0, self.N_rho - 2, device=self.device, dtype=torch.float32).view(-1, 1)
        laplacian = calculate_laplace(phi, self.dh, idx_i)
        return torch.sum((laplacian + self.f[0:-1, 1:-1]) ** 2) / (self.N_rho * (2 * self.N_z + 1))

    def inject_full_guess(self, phi_full: torch.Tensor) -> None:
        with torch.no_grad():
            self.phi_inner.copy_(phi_full[0:-1, 1:-1].to(self.device))

    def iterate(self) -> float:
        if self.optimizer_config.name == "lbfgs":
            def closure() -> torch.Tensor:
                self.optimizer.zero_grad()
                loss = self.compute_loss()
                loss.backward()
                return loss

            self.optimizer.step(closure)
            with torch.no_grad():
                loss = self.compute_loss()
            return float(loss.item())

        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def solve(self, steps: int, verbose: bool = True) -> tuple[list[float], float]:
        history: list[float] = []
        start_time = time.perf_counter()
        for step in range(steps):
            loss = self.iterate()
            history.append(loss)
            if verbose:
                print(f"{step + 1:4d}/{steps:4d}  loss={loss:.8e}")
        runtime_s = time.perf_counter() - start_time
        return history, runtime_s

    def get_phi(self) -> torch.Tensor:
        with torch.no_grad():
            return self.build_phi().detach().clone()


def interpolate_field(phi: torch.Tensor, target_grid: GridConfig) -> torch.Tensor:
    phi_4d = phi.view(1, 1, phi.shape[0], phi.shape[1])
    upsampled = F.interpolate(
        phi_4d,
        size=(target_grid.N_rho, 2 * target_grid.N_z + 1),
        mode="bicubic",
        align_corners=True,
    )
    return upsampled.squeeze(0).squeeze(0)


def summarize_run(
    group: str,
    label: str,
    optimizer: str,
    init_mode: str,
    steps: int,
    runtime_s: float,
    phi: torch.Tensor,
    final_loss: float,
    reference_phi: torch.Tensor,
    note: str = "",
) -> ExperimentResult:
    rel_error = float((torch.norm(phi - reference_phi) / torch.norm(reference_phi)).item())
    return ExperimentResult(
        group=group,
        label=label,
        optimizer=optimizer,
        init_mode=init_mode,
        steps=steps,
        runtime_s=runtime_s,
        final_loss=final_loss,
        rel_error=rel_error,
        note=note,
    )


def run_single_experiment(
    grid: GridConfig,
    optimizer_config: OptimizerConfig,
    init_mode: str,
    device: torch.device,
    reference_phi: torch.Tensor,
    seed: int = 1234,
    injected_guess: torch.Tensor | None = None,
    verbose: bool = False,
) -> tuple[Solver, ExperimentResult]:
    solver = Solver(grid, device, optimizer_config, init_mode=init_mode, seed=seed)
    if injected_guess is not None:
        solver.inject_full_guess(injected_guess)
    history, runtime_s = solver.solve(optimizer_config.steps, verbose=verbose)
    phi = solver.get_phi()
    result = summarize_run(
        group="",
        label="",
        optimizer=optimizer_config.name,
        init_mode=init_mode,
        steps=optimizer_config.steps,
        runtime_s=runtime_s,
        phi=phi,
        final_loss=history[-1],
        reference_phi=reference_phi,
    )
    return solver, result


def run_coarse_to_fine(
    coarse_grid: GridConfig,
    fine_grid: GridConfig,
    device: torch.device,
    coarse_optimizer: OptimizerConfig,
    fine_optimizer: OptimizerConfig,
    reference_phi: torch.Tensor,
    verbose: bool = False,
) -> tuple[Solver, ExperimentResult]:
    coarse_solver = Solver(coarse_grid, device, coarse_optimizer, init_mode="sphere")
    coarse_history, coarse_runtime = coarse_solver.solve(coarse_optimizer.steps, verbose=verbose)
    phi_coarse = coarse_solver.get_phi()
    phi_fine_guess = interpolate_field(phi_coarse, fine_grid)

    fine_solver = Solver(fine_grid, device, fine_optimizer, init_mode="sphere")
    fine_solver.inject_full_guess(phi_fine_guess)
    fine_history, fine_runtime = fine_solver.solve(fine_optimizer.steps, verbose=verbose)
    phi_fine = fine_solver.get_phi()

    result = summarize_run(
        group="",
        label="coarse_to_fine",
        optimizer="lbfgs",
        init_mode="coarse_to_fine",
        steps=coarse_optimizer.steps + fine_optimizer.steps,
        runtime_s=coarse_runtime + fine_runtime,
        phi=phi_fine,
        final_loss=fine_history[-1],
        reference_phi=reference_phi,
        note=(
            f"coarse_loss={coarse_history[-1]:.3e}, fine_start_from_bicubic_interp"
        ),
    )
    return fine_solver, result


def format_markdown_table(results: list[ExperimentResult]) -> str:
    header = [
        "| 方法 | 优化器 | 初值 | 步数 | 最终 loss | 相对误差 | 用时 / s | 备注 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    rows = []
    for item in results:
        rows.append(
            "| {label} | {optimizer} | {init_mode} | {steps} | {loss:.3e} | {rel:.3e} | {runtime:.2f} | {note} |".format(
                label=item.label,
                optimizer=item.optimizer,
                init_mode=item.init_mode,
                steps=item.steps,
                loss=item.final_loss,
                rel=item.rel_error,
                runtime=item.runtime_s,
                note=item.note or "-",
            )
        )
    return "\n".join(header + rows)


def print_result_block(title: str, results: list[ExperimentResult]) -> None:
    print(f"\n=== {title} ===")
    for item in results:
        print(
            f"{item.label:>16s}  loss={item.final_loss:.6e}  rel_error={item.rel_error:.6e}  "
            f"time={item.runtime_s:.2f}s"
        )


def plot_potential(phi: torch.Tensor, rho_2d: torch.Tensor, z_2d: torch.Tensor, rho_1d: torch.Tensor, z_1d: torch.Tensor, output_path: Path) -> None:
    phi_np = phi.cpu().numpy()
    rho_np = rho_2d.cpu().numpy()
    z_np = z_2d.cpu().numpy()
    rho_1d_np = rho_1d.cpu().numpy()
    z_1d_np = z_1d.cpu().numpy()
    theta = np.linspace(0.0, 2.0 * np.pi, 181)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0])

    ax2d = fig.add_subplot(grid[:, 0])
    mesh = ax2d.pcolormesh(z_np, rho_np, phi_np, shading="auto", cmap="viridis")
    ax2d.set_title("Potential in rho-z plane")
    ax2d.set_xlabel("z")
    ax2d.set_ylabel("rho")
    fig.colorbar(mesh, ax=ax2d, label="phi")

    slice_indices = np.linspace(len(z_1d_np) // 2, len(z_1d_np) - 1, 4, dtype=int)
    polar_axes = []
    polar_mesh = None
    vmin = float(phi_np.min())
    vmax = float(phi_np.max())

    for plot_index, z_index in enumerate(slice_indices):
        row = plot_index // 2
        col = 1 + plot_index % 2
        ax = fig.add_subplot(grid[row, col], projection="polar")
        phi_slice = np.repeat(phi_np[:, z_index : z_index + 1], theta.size, axis=1)
        theta_grid, rho_grid = np.meshgrid(theta, rho_1d_np)
        polar_mesh = ax.pcolormesh(
            theta_grid,
            rho_grid,
            phi_slice,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"z = {z_1d_np[z_index]:.2f}")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        polar_axes.append(ax)

    if polar_mesh is not None:
        fig.colorbar(polar_mesh, ax=polar_axes, label="phi", shrink=0.9)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_all_experiments(device: torch.device, verbose: bool = False) -> dict[str, object]:
    coarse_grid = GridConfig(N_rho=40, N_z=40, dh=1.0)
    fine_grid = GridConfig(N_rho=200, N_z=200, dh=0.2)

    reference_coarse = OptimizerConfig(name="lbfgs", lr=1.0, steps=30, lbfgs_max_iter=20, history_size=80)
    reference_fine = OptimizerConfig(name="lbfgs", lr=1.0, steps=50, lbfgs_max_iter=20, history_size=100)

    print("\n--- Building reference solution with coarse-to-fine L-BFGS ---")
    reference_solver = Solver(coarse_grid, device, reference_coarse, init_mode="sphere")
    reference_solver.solve(reference_coarse.steps, verbose=verbose)
    reference_guess = interpolate_field(reference_solver.get_phi(), fine_grid)
    reference_fine_solver = Solver(fine_grid, device, reference_fine, init_mode="sphere")
    reference_fine_solver.inject_full_guess(reference_guess)
    reference_history, reference_runtime = reference_fine_solver.solve(reference_fine.steps, verbose=verbose)
    reference_phi = reference_fine_solver.get_phi()
    reference_loss = reference_history[-1]
    plot_potential(
        reference_phi,
        reference_fine_solver.rho_2d,
        reference_fine_solver.z_2d,
        reference_fine_solver.rho_1d,
        reference_fine_solver.z_1d,
        PLOT_PATH,
    )

    optimizer_results: list[ExperimentResult] = []
    optimizer_setups = [
        ("sgd", OptimizerConfig(name="sgd", lr=5.0e-3, steps=3000)),
        ("adam", OptimizerConfig(name="adam", lr=1.0e-2, steps=2000)),
        ("lbfgs", OptimizerConfig(name="lbfgs", lr=1.0, steps=35, lbfgs_max_iter=20, history_size=80)),
    ]

    print("\n--- Optimizer comparison ---")
    for label, config in optimizer_setups:
        solver, result = run_single_experiment(
            grid=fine_grid,
            optimizer_config=config,
            init_mode="sphere",
            device=device,
            reference_phi=reference_phi,
            seed=1234,
            verbose=verbose,
        )
        result.group = "optimizer"
        result.label = label
        result.note = "same smooth sphere init, comparable CPU budget"
        optimizer_results.append(result)
        del solver

    init_results: list[ExperimentResult] = []
    init_optimizer = OptimizerConfig(name="lbfgs", lr=1.0, steps=35, lbfgs_max_iter=20, history_size=80)
    init_modes = ["zeros", "randn", "sphere"]

    print("\n--- Initialization comparison ---")
    for init_mode in init_modes:
        solver, result = run_single_experiment(
            grid=fine_grid,
            optimizer_config=init_optimizer,
            init_mode=init_mode,
            device=device,
            reference_phi=reference_phi,
            seed=2026,
            verbose=verbose,
        )
        result.group = "initialization"
        result.label = init_mode
        init_results.append(result)
        del solver

    multiscale_coarse = OptimizerConfig(name="lbfgs", lr=1.0, steps=20, lbfgs_max_iter=20, history_size=60)
    multiscale_fine = OptimizerConfig(name="lbfgs", lr=1.0, steps=18, lbfgs_max_iter=20, history_size=80)
    _, multiscale_result = run_coarse_to_fine(
        coarse_grid=coarse_grid,
        fine_grid=fine_grid,
        device=device,
        coarse_optimizer=multiscale_coarse,
        fine_optimizer=multiscale_fine,
        reference_phi=reference_phi,
        verbose=verbose,
    )
    multiscale_result.group = "initialization"
    init_results.append(multiscale_result)

    print_result_block("Optimizer Comparison", optimizer_results)
    print_result_block("Initialization Comparison", init_results)

    payload = {
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "grid": {
            "coarse": asdict(coarse_grid),
            "fine": asdict(fine_grid),
        },
        "reference": {
            "method": "coarse_to_fine_lbfgs",
            "runtime_s": reference_runtime,
            "final_loss": reference_loss,
            "plot_path": str(PLOT_PATH),
        },
        "optimizer_comparison": [asdict(item) for item in optimizer_results],
        "initialization_comparison": [asdict(item) for item in init_results],
        "optimizer_table_markdown": format_markdown_table(optimizer_results),
        "initialization_table_markdown": format_markdown_table(init_results),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return payload


def run_best_only(device: torch.device, verbose: bool = True) -> None:
    coarse_grid = GridConfig(N_rho=40, N_z=40, dh=1.0)
    fine_grid = GridConfig(N_rho=200, N_z=200, dh=0.2)
    coarse_optimizer = OptimizerConfig(name="lbfgs", lr=1.0, steps=35, lbfgs_max_iter=20, history_size=80)
    fine_optimizer = OptimizerConfig(name="lbfgs", lr=1.0, steps=40, lbfgs_max_iter=20, history_size=100)

    print("\n--- Phase 1: Coarse Grid Solving ---")
    coarse_solver = Solver(coarse_grid, device, coarse_optimizer, init_mode="sphere")
    coarse_solver.solve(coarse_optimizer.steps, verbose=verbose)
    phi_coarse = coarse_solver.get_phi()

    print("\n--- Phase 2: Upsampling Physical Field ---")
    phi_fine_guess = interpolate_field(phi_coarse, fine_grid)

    print("\n--- Phase 3: Fine Grid Polishing ---")
    fine_solver = Solver(fine_grid, device, fine_optimizer, init_mode="sphere")
    fine_solver.inject_full_guess(phi_fine_guess)
    fine_solver.solve(fine_optimizer.steps, verbose=verbose)

    plot_potential(
        fine_solver.get_phi(),
        fine_solver.rho_2d,
        fine_solver.z_2d,
        fine_solver.rho_1d,
        fine_solver.z_1d,
        PLOT_PATH,
    )
    print(f"saved figure: {PLOT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the Poisson equation and run comparison experiments.")
    parser.add_argument(
        "--mode",
        choices=["experiments", "best"],
        default="experiments",
        help="`experiments` runs all comparison experiments; `best` only runs the current best coarse-to-fine solver.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step losses during optimization.",
    )
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")

    if args.mode == "best":
        run_best_only(device=device, verbose=args.verbose)
        return

    payload = run_all_experiments(device=device, verbose=args.verbose)
    print(f"\nSaved experiment summary: {RESULTS_PATH}")
    print(f"Saved best-solution figure: {PLOT_PATH}")
    print(f"Reference final loss: {payload['reference']['final_loss']:.6e}")


if __name__ == "__main__":
    main()

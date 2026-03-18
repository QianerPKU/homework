import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def calculate_Q(f, dh, rho):
    return torch.sum(f * rho).item() * 2 * math.pi * (dh ** 2)

def calculate_laplace(phi, dh, idx_i):
    inv_idx_i = 1.0 / (2.0 * idx_i)
    laplacian_inner = 1.0/(dh**2) * ((1.0+inv_idx_i) * phi[2:, 1:-1] + (1.0-inv_idx_i) * phi[0:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, 0:-2] - 4.0 * phi[1:-1, 1:-1])
    laplacian_rho_0 = 1.0/(dh**2) * (4.0*phi[1:2,1:-1] + phi[0:1,2:] + phi[0:1,0:-2] - 6.0*phi[0:1,1:-1])
    return torch.cat([laplacian_rho_0,laplacian_inner], dim=0)

class Solver:
    def __init__(self, N_rho, N_z, dh, device, lr):
        self.N_rho = N_rho
        self.N_z = N_z
        self.dh = dh
        self.device = device
        self.rho_1d = torch.arange(0,N_rho,device=device,dtype=torch.float32) * dh
        self.z_1d = torch.arange(-N_z, N_z+1, device=device, dtype=torch.float32) * dh
        self.rho_2d, self.z_2d = torch.meshgrid(self.rho_1d, self.z_1d, indexing='ij')
        self.eps = 1e-15
        self.r = torch.sqrt(self.rho_2d ** 2 + self.z_2d ** 2) + self.eps
        self.Y = 0.25 * math.sqrt(5.0/math.pi) * (3.0 * ((self.z_2d / self.r) ** 2) - 1.0 )
        self.r0 = 10.0 * (1.0 + self.Y)
        self.f = (0.8 / (1.0 + torch.exp((self.r - self.r0)/0.7))).detach()

        # ---------------- 替换开始 ----------------
        self.Q_total = calculate_Q(self.f, dh, self.rho_2d)
        
        # 构造均匀带电球体的完美解析解作为 Initial Guess
        R_core = 12.0
        
        # 球外电势 (单极子衰减)
        phi_out = self.Q_total / (4.0 * math.pi * self.r)
        
        # 球内电势 (平滑抛物顶，完美消除 1/r 奇异点)
        phi_in = (self.Q_total / (4.0 * math.pi * R_core)) * 0.5 * (3.0 - (self.r / R_core)**2)
        
        # 按照 r 与 R_core 的关系拼接全空间电势场
        phi_guess = torch.where(self.r <= R_core, phi_in, phi_out).detach()

        # 边界依然严格使用这个场
        self.phi_boundary = phi_guess.clone()
        
        # 内部张量直接从这个完美的解析场开始优化！
        self.phi_inner = nn.Parameter(phi_guess[0:-1, 1:-1].clone())

        self.optimizer = torch.optim.LBFGS(
            [self.phi_inner],
            lr=lr,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe',
        )
        # ---------------- 替换结束 ----------------

    def build_phi(self):
        phi = self.phi_boundary.clone()
        phi[0:-1, 1:-1] = self.phi_inner
        return phi

    def compute_loss(self):
        phi = self.build_phi()
        idx_i = 1 + torch.arange(0,self.N_rho - 2,device=self.device,dtype=torch.float32).view(-1,1)
        laplacian = calculate_laplace(phi, self.dh, idx_i)
        return torch.sum((laplacian + self.f[0:-1, 1:-1]) ** 2) / (self.N_rho * (2 * self.N_z + 1))

    def iterate(self):
        def closure():
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            return loss

        self.optimizer.step(closure)

        with torch.no_grad():
            loss = self.compute_loss()

        return loss
    
    def solve(self, steps):
        for i in range(steps):
            loss = self.iterate()
            print(f"{i+1}/{steps}          loss:{loss.item()}")

    def get_phi(self):
        with torch.no_grad():
            return self.build_phi().detach()


def plot_potential(solver, output_path="potential.png"):
    phi = solver.get_phi().cpu().numpy()
    rho = solver.rho_2d.cpu().numpy()
    z = solver.z_2d.cpu().numpy()
    rho_1d = solver.rho_1d.cpu().numpy()
    z_1d = solver.z_1d.cpu().numpy()
    theta = np.linspace(0.0, 2.0 * np.pi, 181)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0])

    ax2d = fig.add_subplot(grid[:, 0])
    mesh = ax2d.pcolormesh(z, rho, phi, shading='auto', cmap='viridis')
    ax2d.set_title('Potential in rho-z plane')
    ax2d.set_xlabel('z')
    ax2d.set_ylabel('rho')
    fig.colorbar(mesh, ax=ax2d, label='phi')

    slice_indices = np.linspace(solver.N_z, 2 * solver.N_z, 4, dtype=int)
    polar_axes = []
    polar_mesh = None
    vmin = float(phi.min())
    vmax = float(phi.max())

    for plot_index, z_index in enumerate(slice_indices):
        row = plot_index // 2
        col = 1 + plot_index % 2
        ax = fig.add_subplot(grid[row, col], projection='polar')
        phi_slice = np.repeat(phi[:, z_index:z_index + 1], theta.size, axis=1)
        theta_grid, rho_grid = np.meshgrid(theta, rho_1d)
        polar_mesh = ax.pcolormesh(
            theta_grid,
            rho_grid,
            phi_slice,
            shading='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f'z = {z_1d[z_index]:.2f}')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        polar_axes.append(ax)

    if polar_mesh is not None:
        fig.colorbar(polar_mesh, ax=polar_axes, label='phi', shrink=0.9)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # 保持物理空间大小绝对一致： rho_max = 200 fm, z_max = 100 fm
    
    # ==========================================
    # 第一阶段：在粗网格上极速求解 (dh = 2.0)
    # ==========================================
    print("\n--- Phase 1: Coarse Grid Solving ---")
    N_rho_coarse = 40
    N_z_coarse = 40
    dh_coarse = 1.0
    
    # 粗网格条件数极小，收敛如闪电
    solver_coarse = Solver(N_rho_coarse, N_z_coarse, dh_coarse, device, lr=1.0)
    solver_coarse.solve(steps=50) # 50步足以在粗网格上砸到底
    
    # 获取粗网格的全局收敛解，形状为 (100, 101)
    phi_coarse = solver_coarse.get_phi()
    
    # ==========================================
    # 第二阶段：物理场升维插值 (Upsampling)
    # ==========================================
    print("\n--- Phase 2: Upsampling Physical Field ---")
    N_rho_fine = 200
    N_z_fine = 200
    dh_fine = 0.2
    
    # 将粗网格解 reshape 为图像张量 [Batch, Channel, Height, Width]
    phi_coarse_tensor = phi_coarse.view(1, 1, N_rho_coarse, 2 * N_z_coarse + 1)
    
    # 使用双三次插值 (Bicubic) 进行平滑放大到细网格尺寸 (400, 401)
    phi_fine_guess = F.interpolate(
        phi_coarse_tensor, 
        size=(N_rho_fine, 2 * N_z_fine + 1), 
        mode='bicubic', 
        align_corners=True
    ).squeeze() # 去除 Batch 和 Channel 维度
    
    # ==========================================
    # 第三阶段：在细网格上进行完美初值微调
    # ==========================================
    print("\n--- Phase 3: Fine Grid Polishing ---")
    solver_fine = Solver(N_rho_fine, N_z_fine, dh_fine, device, lr=1.0)
    
    # 魔法发生的地方：将插值后的完美物理场强行注入细网格的内部节点！
    with torch.no_grad():
        solver_fine.phi_inner.copy_(phi_fine_guess[0:-1, 1:-1])
        
    # 此时细网格的初值已经包含了极其精确的四极形变和表面弥散信息
    # L-BFGS 只需要处理极微小的插值误差，通常 10~20 步就能达到惊人的精度
    solver_fine.solve(steps=100) 
    
    # 出图
    plot_potential(solver_fine)
    print("saved figure: potential.png")

if __name__ == "__main__":
    main()
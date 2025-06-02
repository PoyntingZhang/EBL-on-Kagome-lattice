import numpy as np
from pythtb import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ========================================
# 相分类函数
# ========================================
def classify_phase(t1, t2, t3, nk=51):
    lat = [[4.0, 0.0], [2.0, 2 * np.sqrt(3)]]
    orb = [[0.0, 0.0], [0.25, -0.25], [0.25, 0.0], [0.0, 0.5], [-0.25, 0.75], [-0.25, 0.5]]
    model = tb_model(2, 2, lat, orb)
    model.set_onsite([0.0] * 6)

    model.set_hop(t1, 0, 1, [0, 0]), model.set_hop(t1, 1, 2, [0, 0])
    model.set_hop(t1, 2, 0, [0, 0]), model.set_hop(t1, 5, 3, [0, 0])
    model.set_hop(t1, 3, 4, [0, 0]), model.set_hop(t1, 4, 5, [0, 0])
    model.set_hop(t2, 0, 5, [0, 0]), model.set_hop(t2, 2, 3, [0, 0])
    model.set_hop(t2, 0, 4, [0, -1]), model.set_hop(t2, 1, 3, [0, -1])
    model.set_hop(t2, 2, 4, [1, -1]), model.set_hop(t2, 1, 5, [1, -1])
    model.set_hop(t3, 0, 3, [0, 0]), model.set_hop(t3, 2, 5, [0, 0])
    model.set_hop(t3, 2, 5, [1, -1]), model.set_hop(t3, 1, 4, [1, -1])
    model.set_hop(t3, 0, 3, [0, -1]), model.set_hop(t3, 1, 4, [0, -1])

    k_grid = [[0.0, 0.0], [0.5, 0.5], [2/3, 1/3]]
    k_vec, _, _ = model.k_path(k_grid, nk, report=False)

    evals = model.solve_all(k_vec)
    min_idx = np.argmin(evals)
    k_min = k_vec[min_idx % len(k_vec)]

    if np.linalg.norm(k_min - [0, 0]) < 0.1:
        return 0
    elif np.linalg.norm(k_min - [0.5, 0.5]) < 0.1:
        return 1
    elif np.linalg.norm(k_min - [2/3, 1/3]) < 0.1:
        return 2
    else:
        return 3

# ========================================
# 生成 t2-t3 相图
# ========================================
def compute_phase_map(t1, steps=50, t_range=(-1, 1)):
    t2_vals = np.linspace(*t_range, steps)
    t3_vals = np.linspace(*t_range, steps)
    data = np.zeros((steps, steps), dtype=int)

    for i, t2 in enumerate(t2_vals):
        for j, t3 in enumerate(t3_vals):
            data[j, i] = classify_phase(t1, t2, t3)
    return t2_vals, t3_vals, data

# ========================================
# 主程序：批量绘图
# ========================================
def plot_all_fixed_t1():
    t1_list = np.round(np.arange(-1, 1.01, 0.1), 2)  # 21 个 t1
    steps = 40  # 分辨率
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for t1 in t1_list:
        print(f"绘制 t1 = {t1:.2f}")
        t2_vals, t3_vals, data = compute_phase_map(t1, steps=steps)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(data, origin='lower', extent=[-1, 1, -1, 1],
                       cmap=cmap, norm=norm, aspect='auto')

        ax.set_title(rf"Phase Diagram at $\tau_1$ = {t1:.2f}", fontsize=15)
        ax.set_xlabel(r"$\tau_2$", fontsize=13)
        ax.set_ylabel(r"$\tau_3$", fontsize=13)

        from matplotlib.patches import Patch
        labels = ['Γ-phase', 'M-phase', 'K-phase', 'Other']
        handles = [Patch(color=colors[i], label=labels[i]) for i in range(4)]
        ax.legend(handles=handles, loc='upper right')

        plt.tight_layout()
        plt.show()

# ========================================
# 运行主程序
# ========================================
if __name__ == '__main__':
    plot_all_fixed_t1()

import numpy as np
from matplotlib import pyplot as plt
from pythtb import *

theta = np.pi / 6
t1 = 0.5 * np.exp(1j * theta)
t1_c = t1.conjugate()
t2 = -0.7
t3 = 0.5


# 创建 tight-binding 模型函数
def create_model(t1, t2, t3, delta):
    lat = [[2.0, -2 * np.sqrt(3)], [2.0, 2 * np.sqrt(3)]]
    orb = [[0, -1 / 4], [1 / 4, -1 / 4], [1 / 4, 0.0], [0.0, 1 / 4], [-1 / 4, 1 / 4], [-1 / 4, 0]]
    model = tb_model(2, 2, lat, orb)
    model.set_onsite([delta] * 3 + [-delta] * 3)

    # 最近邻
    model.set_hop(t1_c, 0, 1, [0, 0])
    model.set_hop(t1_c, 1, 2, [0, 0])
    model.set_hop(t1_c, 2, 0, [0, 0])
    model.set_hop(t1, 5, 3, [0, 0])
    model.set_hop(t1, 3, 4, [0, 0])
    model.set_hop(t1, 4, 5, [0, 0])

    # 次近邻
    model.set_hop(t2, 0, 5, [0, 0])
    model.set_hop(t2, 2, 3, [0, 0])
    model.set_hop(t2, 2, 4, [1, 0])
    model.set_hop(t2, 1, 5, [1, 0])
    model.set_hop(t2, 3, 1, [0, 1])
    model.set_hop(t2, 4, 0, [0, 1])

    # 长程
    model.set_hop(t3, 0, 3, [0, 0])
    model.set_hop(t3, 2, 5, [0, 0])
    model.set_hop(t3, 1, 4, [1, 0])
    model.set_hop(t3, 2, 5, [1, 0])
    model.set_hop(t3, 4, 1, [0, 1])
    model.set_hop(t3, 3, 0, [0, 1])
    return model


# 设置子图
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
deltas = [1, -1]

for i, delta in enumerate(deltas):
    model = create_model(t1, t2, t3, delta)
    wf = wf_array(model, [100, 100])
    wf.solve_on_grid([-0.5, -0.5])
    plaq = wf.berry_flux([0], individual_phases=True)

    im = axs[i].imshow(plaq.T, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), cmap="RdBu_r")
    axs[i].set_title(rf"$\delta = {delta}$", fontsize=12)
    axs[i].set_xlabel(r"$k_x$")
    axs[i].set_ylabel(r"$k_y$")

    # 每个子图独立 colorbar
    cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)




# 添加统一颜色条



# 添加底部注释
param_str = rf"$t_1 = 0.5e^{{i\pi/6}},\quad t_2 = {t2},\quad t_3 = {t3}$"
fig.text(0.5, 0.05,  param_str, ha='center', fontsize=10)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("berry_curvature_delta_scan.pdf", dpi=300)
plt.show()





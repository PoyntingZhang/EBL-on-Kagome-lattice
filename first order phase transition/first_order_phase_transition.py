from pythtb import *  # tight-binding toolkit
import numpy as np
import matplotlib.pyplot as plt


# 构建 TB 模型的函数
def create_model(t1, t2, t3, delta=0.0):
    lat = [[2.0, -2 * np.sqrt(3)], [2.0, 2 * np.sqrt(3)]]
    orb = [[0, -1 / 4], [1 / 4, -1 / 4], [1 / 4, 0.0],
           [0.0, 1 / 4], [-1 / 4, 1 / 4], [-1 / 4, 0]]

    model = tb_model(2, 2, lat, orb)
    model.set_onsite([delta] * 6)

    # 最近邻 t1
    model.set_hop(t1, 0, 1, [0, 0])
    model.set_hop(t1, 1, 2, [0, 0])
    model.set_hop(t1, 2, 0, [0, 0])
    model.set_hop(t1, 5, 3, [0, 0])
    model.set_hop(t1, 3, 4, [0, 0])
    model.set_hop(t1, 4, 5, [0, 0])

    # 次近邻 t2
    model.set_hop(t2, 0, 5, [0, 0])
    model.set_hop(t2, 2, 3, [0, 0])
    model.set_hop(t2, 2, 4, [1, 0])
    model.set_hop(t2, 1, 5, [1, 0])
    model.set_hop(t2, 3, 1, [0, 1])
    model.set_hop(t2, 4, 0, [0, 1])

    # 长程 t3
    model.set_hop(t3, 0, 3, [0, 0])
    model.set_hop(t3, 2, 5, [0, 0])
    model.set_hop(t3, 1, 4, [1, 0])
    model.set_hop(t3, 2, 5, [1, 0])
    model.set_hop(t3, 4, 1, [0, 1])
    model.set_hop(t3, 3, 0, [0, 1])

    return model


# k 空间路径设置
path = [[0., 0.], [1 / 3., 1 / 3.], [0.5, 0.], [0., 0.]]
labels = (r'$\Gamma$', r'$K$', r'$M$', r'$\Gamma$')
nk = 121

# 参数组定义（每组为一张图，每组中三个为 subplot）
parameter_groups = [
    [(-0.5, -0.75, 0.6), (-0.5, -0.75, 0.75), (-0.5, -0.75, 0.9)],
    [(0.0, -0.5, 0.1), (0.0, -0.5, 0.0), (0.0, -0.5, -0.1)],
    [(0.5, 0.75, 0.4), (0.5, 0.75, 0.5), (0.5, 0.75, 0.6)]
]

# 绘图
for group in parameter_groups:
    tau1_val = group[0][0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (t1, t2, t3) in enumerate(group):
        model = create_model(t1, t2, t3)
        k_vec, k_dist, k_node = model.k_path(path, nk)
        evals = model.solve_all(k_vec)

        ax = axs[idx]
        ax.set_xlim(k_node[0], k_node[-1])
        ax.set_xticks(k_node)
        ax.set_xticklabels(labels)
        for x in k_node:
            ax.axvline(x=x, color='k', linewidth=0.5)

        # 子图标题
        ax.set_title(
            rf"$\tau_1$={t1}, $\tau_2$={t2}, $\tau_3$={t3}", fontsize=10
        )
        if idx == 0:
            ax.set_ylabel("Energy")
        ax.set_xlabel("k-path")

        for band in evals:
            ax.plot(k_dist, band)

    # 总标题
    fig.suptitle(
        f"First order phase transition with τ₁ = {tau1_val}, respectively",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"phase_transition_tau1_{tau1_val:+.1f}.pdf")
    plt.show()

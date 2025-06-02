from __future__ import print_function
from pythtb import *
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# initialise parameters
t1_init = -1.0
t2_init = 0.0
t3_init = 1.0
delta = 0.0


# create model function
def create_model(t1, t2, t3):
    # define lattice vectors
    lat = [[4.0, 0.0], [2.0, 2 * np.sqrt(3)]]
    # define orbital positions
    orb = [
        [0.0, 0.0],  # orbital 0
        [1 / 4, -1 / 4],  # 1
        [1 / 4, 0.0],  # 2
        [0.0, 1 / 2],  # 3
        [-1 / 4, 3 / 4],  # 4
        [-1 / 4, 1 / 2]  # 5
    ]

    # create model
    my_model = tb_model(2, 2, lat, orb)

    # set onsite energy
    my_model.set_onsite([delta] * 6)



    # set hoppings
    # nearest neighbour
    my_model.set_hop(t1, 0, 1, [0, 0])
    my_model.set_hop(t1, 1, 2, [0, 0])
    my_model.set_hop(t1, 2, 0, [0, 0])
    my_model.set_hop(t1, 5, 3, [0, 0])
    my_model.set_hop(t1, 3, 4, [0, 0])
    my_model.set_hop(t1, 4, 5, [0, 0])

    # next nearest neighbour
    my_model.set_hop(t2, 0, 5, [0, 0])
    my_model.set_hop(t2, 2, 3, [0, 0])
    my_model.set_hop(t2, 0, 4, [0, -1])
    my_model.set_hop(t2, 1, 3, [0, -1])
    my_model.set_hop(t2, 2, 4, [1, -1])
    my_model.set_hop(t2, 1, 5, [1, -1])

    # next next nearest neighbour
    my_model.set_hop(t3, 0, 3, [0, 0])
    my_model.set_hop(t3, 2, 5, [0, 0])
    my_model.set_hop(t3, 2, 5, [1, -1])
    my_model.set_hop(t3, 1, 4, [1, -1])
    my_model.set_hop(t3, 0, 3, [0, -1])
    my_model.set_hop(t3, 1, 4, [0, -1])

    return my_model


# k path
path = [[0., 0.], [1 / 2, 1 / 2], [2 / 3, 1 / 3], [0, 0]]
label = (r'$\Gamma $', r'$K$', r'$M$', r'$\Gamma $')
nk = 121
k_vec, k_dist, k_node = create_model(t1_init, t2_init, t3_init).k_path(path, nk)

# figure settings
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

# initialise bands
lines = [ax.plot(k_dist, np.zeros(len(k_dist)))[0] for _ in range(6)]
ax.set_ylim(-4, 4)
ax.set_xticks(k_node)
ax.set_xticklabels(label, fontsize=12)
for xc in k_node:
    ax.axvline(x=xc, color='gray', linestyle='--', linewidth=0.5)


# make sliders
slider_params = [
    {'pos': [0.15, 0.15, 0.7, 0.03], 'label': 't₁', 'valinit': t1_init, 'valmin': -1.0, 'valmax': 1.0},
    {'pos': [0.15, 0.10, 0.7, 0.03], 'label': 't₂', 'valinit': t2_init, 'valmin': -1.0, 'valmax': 1.0},
    {'pos': [0.15, 0.05, 0.7, 0.03], 'label': 't₃', 'valinit': t3_init, 'valmin': -1.0, 'valmax': 1.0}
]

sliders = []
for param in slider_params:
    ax_slider = plt.axes(param['pos'])
    slider = Slider(
        ax=ax_slider,
        label=param['label'],
        valmin=param['valmin'],
        valmax=param['valmax'],
        valinit=param['valinit'],
        # 可选的美化参数
        initcolor='none',
        track_color='lightgrey',
        handle_style={'facecolor': 'steelblue', 'edgecolor': 'white', 'size': 8}
    )
    sliders.append(slider)


# 更新函数（添加计算保护）
def update(_):
    try:
        params = [s.val for s in sliders]
        model = create_model(*params)
        evals = model.solve_all(k_vec)

        # 更新所有能带曲线
        for i, line in enumerate(lines):
            line.set_ydata(evals[i])
            line.set_color(plt.cm.tab10(i))  # 添加颜色区分

        # 动态调整y轴范围（可选）
        current_ymax = np.max(np.abs(evals))
        ax.set_ylim(-current_ymax * 1.1, current_ymax * 1.1)

        fig.canvas.draw_idle()
    except Exception as e:
        print(f"Error updating bands: {str(e)}")


# 绑定事件
for slider in sliders:
    slider.on_changed(update)

# 初始计算
update(None)

plt.show()


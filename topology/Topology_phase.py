import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Ruby_Chern import compute_chern_numbers
from matplotlib.colors import TwoSlopeNorm

# 参数扫描范围
theta_list = np.linspace(-np.pi, np.pi, 75)     # θ 从 -π 到 π
delta_list = np.linspace(-3, 3, 75)             # δ 从 -3 到 3

# 固定参数
t = 0.5
t2 = -0.7
t3 = 0.5

# 存储 Chern 数
chern_map = np.zeros((len(delta_list), len(theta_list)))

# 计算
for i, delta in enumerate(tqdm(delta_list, desc="计算 Chern 数")):
    for j, theta in enumerate(theta_list):
        chern = compute_chern_numbers(t, theta, t2, t3, delta=delta)
        chern_map[i, j] = np.round(chern.real, 4)  # 仅记录最低能带

# 可视化
fig, ax = plt.subplots(figsize=(8,6))

# 使用更美观的色图，并设置颜色以0为中心
norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
c = ax.pcolormesh(theta_list, delta_list, chern_map, shading='auto', cmap='coolwarm', norm=norm)

# 坐标轴标签与刻度
ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
ax.set_ylabel(r'$\delta$', fontsize=12)
ax.set_title('Chern number of the lowest band', fontsize=14)

# 设置 θ 坐标轴刻度显示为 π 单位
ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# 添加颜色条
fig.colorbar(c, ax=ax, label='Chern number')

plt.tight_layout()

# 保存 PDF 和 PNG，去除白边
plt.savefig("chern_phase_diagram.pdf", format='pdf', bbox_inches='tight')
plt.savefig("chern_phase_diagram.png", dpi=300, bbox_inches='tight')

plt.show()

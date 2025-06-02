import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 定义晶格常数
a1 = np.array([1, 0])
a2 = np.array([0.5, np.sqrt(3) / 2])

# Kagome 晶格中每个原胞的三个原子（以原胞原点为参考）
basis = [
    np.array([0, 0]),
    np.array([0.5, 0]),
    np.array([0.25, np.sqrt(3) / 4])
]

# 键连接关系，(i, j, dx) 表示第 i 个原子连接第 j 个原子，dx 为单位胞偏移
bonds = [
    (0, 1, [0, 0]),
    (1, 2, [0, 0]),
    (2, 0, [0, 0]),
    (0, 1, [-1, 0]),
    (0, 2, [0,-1]),
    (1, 2, [1, -1])
]

fig, ax = plt.subplots(figsize=(8, 8))
Nx, Ny = 7, 7

# 用来保存所有 boson 中心的位置
boson_centers = []

# 遍历所有 unit cell
for ix in range(Nx):
    for iy in range(Ny):
        cell_origin = ix * a1 + iy * a2

        for (i, j, delta) in bonds:
            neighbor_origin = (ix + delta[0]) * a1 + (iy + delta[1]) * a2
            r1 = cell_origin + basis[i]
            r2 = neighbor_origin + basis[j]

            # boson 椭圆
            center = (r1 + r2) / 2
            vec = r2 - r1
            length = np.linalg.norm(vec) * 0.6
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            ellipse = Ellipse(center, width=length, height=0.1, angle=angle,
                              facecolor='red', alpha=0.5, edgecolor='none')
            ax.add_patch(ellipse)

            # 记录 boson 中心
            boson_centers.append(center)

# 根据空间距离连接 boson，形成 ruby 晶格
# 判断两个 boson 中心是否应该相连：若距离小于某阈值就连线
threshold = 0.45 # 根据 ruby 晶格的连接方式微调
boson_centers = np.array(boson_centers)

for i in range(len(boson_centers)):
    for j in range(i + 1, len(boson_centers)):
        dist = np.linalg.norm(boson_centers[i] - boson_centers[j])
        if dist < threshold:
            ax.plot(
                [boson_centers[i][0], boson_centers[j][0]],
                [boson_centers[i][1], boson_centers[j][1]],
                color='blue', linewidth=0.5
            )

# 图形参数
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()

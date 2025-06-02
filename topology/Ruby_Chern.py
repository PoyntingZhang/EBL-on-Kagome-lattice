from pythtb import *
import numpy as np

def compute_chern_numbers(t, theta, t2, t3, delta=0.3, grid=[50, 50]):
    """
    输入:
        t: 最近邻跃迁振幅
        theta: 最近邻跃迁的相位（弧度）
        t2: 次近邻跃迁
        t3: 三近邻跃迁
        delta: 在位能（默认 0）
        grid: k 空间网格大小（默认 [31,31]）
    输出:
        一个长度为6的列表，包含6条能带的 Chern 数
    """

    t1 = t * np.exp(1j * theta)
    t1_c = t1.conjugate()

    # 定义晶格矢量
    lat = [[4.0, 0.0], [2.0, 2 * np.sqrt(3)]]
    # 定义6个轨道的位置
    orb = [
        [0.0, 0.0],
        [1 / 4, -1 / 4],
        [1 / 4, 0.0],
        [0.0, 1 / 2],
        [-1 / 4, 3 / 4],
        [-1 / 4, 1 / 2]
    ]

    # 构建模型
    model = tb_model(2, 2, lat, orb)
    model.set_onsite([delta,delta,delta,-delta,-delta,-delta])

    # t1 跃迁（带相位）
    model.set_hop(t1_c, 0, 1, [0, 0])
    model.set_hop(t1_c, 1, 2, [0, 0])
    model.set_hop(t1_c, 2, 0, [0, 0])
    model.set_hop(t1, 5, 3, [0, 0])
    model.set_hop(t1, 3, 4, [0, 0])
    model.set_hop(t1, 4, 5, [0, 0])

    # t2 次近邻
    model.set_hop(t2, 0, 5, [0, 0])
    model.set_hop(t2, 2, 3, [0, 0])
    model.set_hop(t2, 0, 4, [0, -1])
    model.set_hop(t2, 1, 3, [0, -1])
    model.set_hop(t2, 2, 4, [1, -1])
    model.set_hop(t2, 1, 5, [1, -1])

    # t3 三近邻
    model.set_hop(t3, 0, 3, [0, 0])
    model.set_hop(t3, 2, 5, [0, 0])
    model.set_hop(t3, 2, 5, [1, -1])
    model.set_hop(t3, 1, 4, [1, -1])
    model.set_hop(t3, 0, 3, [0, -1])
    model.set_hop(t3, 1, 4, [0, -1])

    # 构建 Berry 曲率计算对象
    wf = wf_array(model, grid)
    wf.solve_on_grid([1/2, 1/2])
    chern=wf.berry_flux([0])
    return chern





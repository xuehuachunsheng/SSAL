# 模拟信息度量的大小
import numpy as np
import matplotlib
import scienceplots
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np 
plt.style.use('science')
font = {'family' : 'Times New Roman',
        'weight' : 'regular',
        'size'   : 14}
matplotlib.rc('font', **font)

# 生成5个二维高斯分布N
mus = np.array([[0, 0],
                [3, 3],
                [3,-3],
                [-3, 3],
                [-3, -3]])  # 均值
sigmas = np.array([[[1, 0], [0, 1]],
                   [[2, 0.5], [0.5, 1]],
                   [[1, 0.5], [0.5, 1]],
                   [[1, 0.5], [0.5, 1]],
                   [[1, 0.5], [0.5, 1]]])  # 协方差矩阵

Ns = []
for i in range(5):
    Ns.append(np.random.multivariate_normal(mus[i], sigmas[i], size=1000))
# 绘制散布图
plt.scatter(Ns[0][:, 0], Ns[0][:, 1], marker="o", s=7, alpha=0.5)
plt.scatter(Ns[1][:, 0], Ns[1][:, 1], marker="v", s=7, alpha=0.5)
plt.scatter(Ns[2][:, 0], Ns[2][:, 1], marker="+", s=7, alpha=0.5)
plt.scatter(Ns[3][:, 0], Ns[3][:, 1], marker="d", s=7, alpha=0.5)
plt.scatter(Ns[4][:, 0], Ns[4][:, 1], marker="x", s=7, alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Gaussian Distribution')
# 显示图像
plt.show()
plt.savefig("/home/wyx/vscode_projects/SSAL/test/simulation.pdf", dpi=300)


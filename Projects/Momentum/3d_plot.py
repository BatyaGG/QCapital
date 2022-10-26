import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager


DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS


dbm = DBManager(DB_USERNAME, DB_PASS, 'DWH')


params = dbm.select_df('select * from momentum_tuning_new_dots_vola_seventy')
x_col = 'dots_n'
y_col = 'vola_window'
z_col = 'beta'


# Xm = params[x_col]
# Ym = params[y_col]
# Zm = params[z_col].to_numpy().reshape((1, -1))

# print(Zm)


# Xm = np.arange(-5, 5, 0.25)
# Ym = np.arange(-5, 5, 0.25)
# Xm, Ym = np.meshgrid(Xm, Ym)

# R = np.sqrt(Xm**2 + Ym**2)
# Zm = np.sin(R)

# Make data.2
# #372: 190 | 13
# #373: 190 | 14
# #374: 190 | 15
# #375: 190 | 16
# #376: 190 | 17
# #377: 190 | 18
# #378: 190 | 19
# #379: 190 | 20
# #380: 200 | 1
# #381: 200 | 2
# #382: 200 | 3
# #383: 200 | 4
# #384: 200 | 5
# #385: 200 | 6
# #386: 200 | 7
# #387: 200 | 8
# #388: 200 | 9
# #389: 200 | 10
# #390: 200 | 11
# #391: 200 | 12
# #392: 200 | 13
# #393: 200 | 14
X = params[x_col].unique()
Y = params[y_col].unique()
X.sort()
Y.sort()
Xm, Ym = np.meshgrid(X, Y)
Zm = np.zeros(Xm.shape)
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        z = params[(params[x_col]==x) & (params[y_col]==y)]
        if z.shape[0] == 0:
            Zm[j, i] = None
        else:
            Zm[j, i] = z.iloc[0][z_col]

count_nans = 0
for i in range(Zm.shape[0]):
    for j in range(Zm.shape[1]):
        val = Zm[i, j]
        if np.isnan(val):
            count_nans += 1
            neighbrs = []
            if i > 0 and not np.isnan(Zm[i - 1, j]):
                neighbrs.append(Zm[i - 1, j])
            if i < Zm.shape[0] - 1 and not np.isnan(Zm[i + 1, j]):
                neighbrs.append(Zm[i + 1, j])
            if j > 0 and not np.isnan(Zm[i, j - 1]):
                neighbrs.append(Zm[i, j - 1])
            if j < Zm.shape[1] - 1 and not np.isnan(Zm[i, j + 1]):
                neighbrs.append(Zm[i, j + 1])
            Zm[i, j] = sum(neighbrs) / len(neighbrs)

print(count_nans, 'nans found')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(Xm, Ym, Zm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

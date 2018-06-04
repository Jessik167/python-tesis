from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

x = [0.4, 0.5, 0.6, 0.7]
y = [20, 40, 60, 80, 100]
z = [[5.1,5,6.3,7.6],
     [8.1,9.3,11.2,13.5], [10.2,13.4,15.5,18.3],
     [13.2,15.9,19.5,24.8], [16.3,19.1,23.2,28.5]]

x2 = [0.4, 0.5, 0.6, 0.7]
y2 = [20, 40, 60, 80, 100]
z2 = [[4.4,5.5,7.1,8.3],
     [8.6,9.5,12.1,13.6], [10.7,13.5,15.1,19],
     [13,15.6,18.8,24], [15.3,18.7,22.8,27.1]]

x3 = [0.4, 0.5, 0.6, 0.7]
y3 = [20, 40, 60, 80, 100]
z3 = [[5.2,6.5,7.4,8.5],
     [8.6,10.2,12.1,14.3], [11.2,13.6,15.8,19.3],
     [13.5,16.9,20.2,25.2], [16.1,19.7,24,28.8]]


X, Y = np.meshgrid(x, y)
Z = np.array(z)

X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.array(z2)

X3, Y3 = np.meshgrid(x3, y3)
Z3 = np.array(z3)

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='b',alpha=0.5)
ax.plot_surface(X2, Y2, Z2, color='r',alpha=0.4)
ax.plot_surface(X3, Y3, Z3, color='g',alpha=0.3)

plt.xlabel('probabilidad')
plt.ylabel('número de nodos')
ax.set_zlabel('coloreado')
ax.set_title('comparaciones de estrategias')

gris = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
rosa = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
piel = mpl.lines.Line2D([0],[0], linestyle="none", c='g', marker = 'o')


ax.legend([gris,rosa,piel],['smallest last','independent set','AGH'])

plt.show()
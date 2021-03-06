from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

'''
#número cromático
x = [0.4, 0.5, 0.6, 0.7]
y = [20, 40, 60, 80, 100]
z = [[5.4,6.1,7.2,8.6],
     [8.7,10.1,12.3,14.5], [11.4,14.3,16.5,19.5],
     [14.6,17.2,20.6,24.6], [17.1,20,24.2,29.1]]

x2 = [0.4, 0.5, 0.6, 0.7]
y2 = [20, 40, 60, 80, 100]
z2 = [[5.6,6.5,7.9,9.6],
     [9,11.2,13.1,14.8], [11.6,13.4,15.9,19.8],
     [14.3,17.8,20.6,24.7], [16.6,19.6,23.7,28.5]]

x3 = [0.4, 0.5, 0.6, 0.7]
y3 = [20, 40, 60, 80, 100]
z3 = [[5.1,5.4,6.8,8.3],
     [8.2,9.5,11.5,13.5], [10.5,13.1,16,18.5],
     [13.2,16.2,19.3,23.6], [15.4,19.5,23.1,28.2]]

x4 = [0.4, 0.5, 0.6, 0.7]
y4 = [20, 40, 60, 80, 100]
z4 = [[5.3,5.6,7.1,8.5],
     [8.2,9.7,11.5,13.5], [10.5,13.1,15.9,18.5],
     [13.2,16.3,19.3,23.5], [15.4,19.5,23.1,28.2]]
'''
#Tiempo
x = [0.4, 0.5, 0.6, 0.7]
y = [20, 40, 60, 80, 100]
z = [[0.004726434,0.005512142,0.008313179,0.006825376],
     [0.006895781,0.007475185,0.009120369,0.011662245], [0.033510137,0.040618896,0.045164585,0.04819231],
     [0.065345931,0.074110579,0.08299191,0.096085191], [0.099286032,0.116910982,0.1352669,0.153556871]]

x2 = [0.4, 0.5, 0.6, 0.7]
y2 = [20, 40, 60, 80, 100]
z2 = [[0.00352664,0.004609609,0.006759214,0.00581789],
     [0.012036848,0.013295221,0.015041375,0.016791105], [0.028447342,0.033947992,0.037641811,0.04012177],
     [0.056364846,0.06348772,0.069637942,0.080651116], [0.085638094	,0.099723554,0.11531291,0.130744338]]

x3 = [0.4, 0.5, 0.6, 0.7]
y3 = [20, 40, 60, 80, 100]
z3 = [[0.013965201,0.097443795,0.103529,0.034411287],
     [0.03748517,0.070295763,0.048426223,0.056240058], [0.09823904,0.12021718,0.141024518,0.168244886],
     [0.200767803,0.308278131,0.293617463,0.361958742], [0.355150223,0.43852582,0.518617058,0.605478573]]

x4 = [0.4, 0.5, 0.6, 0.7]
y4 = [20, 40, 60, 80, 100]
z4 = [[0.007778454,0.011482477,0.011171055,0.012671518],
     [0.035923576,0.040617132,0.048425412,0.054673672], [0.092170548,0.120284438,0.142216611,0.171536946],
     [0.203558302,0.250222063,0.30069983,0.347164154], [0.355944204	,0.437720585,0.530175185,0.605384135]]

X, Y = np.meshgrid(x, y)
Z = np.array(z)

X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.array(z2)

X3, Y3 = np.meshgrid(x3, y3)
Z3 = np.array(z3)

X4, Y4 = np.meshgrid(x4, y4)
Z4 = np.array(z4)

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='b')#,alpha=0.5)
ax.plot_surface(X2, Y2, Z2, color='r')#,alpha=0.4)
ax.plot_surface(X3, Y3, Z3, color='g')#,alpha=0.3)
ax.plot_surface(X4, Y4, Z4, color='y')#,alpha=0.3)

plt.xlabel('probabilidad')
plt.ylabel('número de nodos')
ax.set_zlabel('Tiempo (Segundos)')
ax.set_title('comparaciones de estrategias (Tiempo)')

gris = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
rosa = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
piel = mpl.lines.Line2D([0],[0], linestyle="none", c='g', marker = 'o')
amarillo = mpl.lines.Line2D([0],[0], linestyle="none", c='y', marker = 'o')

#
ax.legend([gris,rosa,piel,amarillo],['smallest last','independent set','AGH-Metrópolis','AGH-Escalando'])

plt.show()
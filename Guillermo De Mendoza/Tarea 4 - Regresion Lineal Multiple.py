import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math

#set the data
data = np.array([
    [8500, 2, 2800], 
    [4700, 5, 200], 
    [5800, 3, 400], 
    [7400, 2, 500], 
    [6200, 5, 3200], 
    [7300, 3, 1800], 
    [5600, 4, 900]])

#organice the data into dimentions
df = pd.DataFrame(columns=['Quantity_Sold', 'Price', 'Advertising'], data=data)
xdata = df["Quantity_Sold"].tolist()
ydata = df["Price"].tolist()
zdata = df["Advertising"].tolist()

#build the model
preX = []
preY = []
for d in data:
    preX.append([
        1
        ,d[0]
        ,d[1]
    ])
    preY.append(d[2])
    
X = np.array(preX)
Y = np.array(preY)

Xt = np.transpose(X)

Wls =  np.dot(
    np.dot(np.linalg.inv(np.dot(Xt,X)),Xt),
    Y
)

#Formula
print("Z = %f + %f X + %f Y"%(Wls[0],Wls[1],Wls[2]))

#Calculate the predictions
predictions = []
for d in data:
    prediction = Wls[0] + Wls[1]*d[0] + Wls[2]*d[1]
    predictions.append([d[0],d[1],prediction])
    
#Graph
p0 = [0,0,Wls[0] + Wls[1]*0 + Wls[2]*0]
p1 = [9000,0,Wls[0] + Wls[1]*9000 + Wls[2]*0]
p2 = [0,8,Wls[0] + Wls[1]*0 + Wls[2]*8]

x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2
ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]
u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
point  = np.array(p0)
normal = np.array(u_cross_v)
d = -point.dot(normal)
xx, yy = np.meshgrid(range(9000), range(8))
z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt3d.scatter3D(xdata, ydata, zdata, color="red")

#plot points
plt3d.set_xlabel('x')
plt3d.set_ylabel('y')
plt3d.set_zlabel('z');

plt.show()
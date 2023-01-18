import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('accepted.gz', sep = " ", header= None)
xdata = df.iloc[:, 0]
ydata = df.iloc[:, 1]
zdata = df.iloc[:, 2]
tdata = df.iloc[:, 3]

%matplotlib qt 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xdata, ydata, zdata, marker= '.')
plt.savefig('data.png')
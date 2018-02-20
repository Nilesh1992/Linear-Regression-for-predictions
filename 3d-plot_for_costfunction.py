# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:34:47 2018

@author: NILESH
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
import get_learning_data
import compute_cost_function

#Tomorrow's task to draw a surface plot for the cost function
fig = plt.figure()
ax = fig.gca(projection='3d')

object_for_data_set = get_learning_data.ReadTextDataSet()
#****************************Single variable linear regression*******************
file_path = os.getcwd() + "\ex1data1.txt"
traning_input = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,0)
traning_output = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,1)
length = len(traning_input)
new_feature = nu.ones((length,1),dtype=nu.float64)
Y = traning_output
X = nu.append(new_feature,traning_input,axis=1)
# Make data.
theta0 = np.arange(5,-1,-0.25)
theta1 = np.arange(-10,10,0.25)
theta0, theta1 = np.meshgrid(theta0, theta1)
row,col = theta0.shape
J = np.zeros((row,col),dtype=np.float64)
theta = np.zeros((1,2),dtype=np.float64)
for i in range(0,row-1):
    for j in range(0,col-1):
          theta[0][0] = theta0[i][j]
          theta[0][1] = theta1[i][j]
          J[i][j] = compute_cost_function.compute_cost_function_for_data(X,theta,Y)
ax.plot_surface(theta0, theta1, J, cmap=cm.coolwarm,linewidth=0, antialiased=False) #This generates the cost fuction suface map
ax.contour(theta0, theta1, J, cmap=cm.coolwarm)#This generates contour map          
          
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:27:55 2018

@author: NILESH
"""
import os
import get_learning_data
import numpy as nu
import gradient_descent
import matplotlib.pyplot as plt

def y_intercept(x_value):
    return ((1.16636235*x_value)-3.63029144)

object_for_data_set = get_learning_data.ReadTextDataSet()
#****************************Single variable linear regression*******************
file_path = os.getcwd() + "\ex1data1.txt"
traning_input = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,0)
traning_output = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,1)
object_for_data_set.visualize_one_variable_data(traning_input,traning_output)
length = len(traning_input)
new_feature = nu.ones((length,1),dtype=nu.float64)
Y = traning_output
X = nu.append(new_feature,traning_input,axis=1)
number_of_parameters = len(X[0])
theta = nu.zeros((1,number_of_parameters),dtype=nu.float64)
number_of_iteration = 1500
learning_rate = 0.01
J_history,theta = gradient_descent.gradient_descent_for_function_minimization(number_of_iteration,learning_rate,X,theta,Y)   
x = nu.arange(5,25,0.01)
y = y_intercept(x)
plt.plot(x,y_intercept(x),'r')
#******************************Multiple variable linear regression******************

    
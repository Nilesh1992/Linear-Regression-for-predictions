# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:44:50 2018

@author: NILESH
"""
import numpy as nu
import compute_cost_function
def gradient_descent_for_function_minimization(iteration,learning_rate,X,theta,Y):
    J_history = nu.zeros((iteration,1),dtype=nu.float64)
    m = len(X)
    for i in range(0,iteration):
        output_vertor = (nu.matmul(X,nu.transpose(theta)) - Y)
        derivative_with_respect_theta = (1/m) * nu.matmul(nu.transpose(X),output_vertor)
        theta = theta - (learning_rate*(nu.transpose(derivative_with_respect_theta)))
        J_history[i,0] = compute_cost_function.compute_cost_function_for_data(X,theta,Y)
    return J_history,theta    
        
        
    
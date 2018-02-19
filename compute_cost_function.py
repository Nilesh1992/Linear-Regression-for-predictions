# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:50:58 2018

@author: NILESH
"""
import numpy
#Theta should be 1*n vector
def compute_cost_function_for_data(X,theta,Y):    
    try:
        m = len(X)
        output_vertor = (numpy.matmul(X,numpy.transpose(theta)) - Y)
        cost = (1/(2*m))*(numpy.matmul(numpy.transpose(output_vertor),output_vertor))
    except Exception:
        print("Some issue while calculating cost function")
        

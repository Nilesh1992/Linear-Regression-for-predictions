# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:49:41 2018

@author: NILESH
"""
#import pandas 
#import re
import os
import numpy
import matplotlib.pyplot as plt

class  ReadTextDataSet:
    #Input: File_path of the text file
    #low_high: The column which sepreate input from output data ex: [1,2,3] so index = 2 
    #seprates from input to actual output 
    #input_or_output: 0-input data and 1-actual data
    def get_data_in_range_from_the_data_set(self,file_path,low_high,input_or_output=0):
        file_handler = open(file_path,'r')
        triplets=file_handler.read().split()
        for i in range(0,len(triplets)): triplets[i]=triplets[i].split(',')
        formated_data = numpy.array(triplets, dtype=numpy.float64)
        row_size = len(formated_data[0])
        col_size = len(formated_data)
        learing_data_matrix = numpy.reshape(formated_data,(col_size,row_size))
        if (input_or_output==0):
             learing_data_matrix = learing_data_matrix[:,0:low_high]
        else:
            learing_data_matrix = learing_data_matrix[:,low_high:row_size]
        return learing_data_matrix        
        
    def visualize_one_variable_data(self,data_x_axis,y_axis):
        plt.plot(data_x_axis,y_axis,'go')
        

object_for_data_set = ReadTextDataSet()
file_path = os.getcwd() + "\ex1data1.txt"
traning_input = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,0)
traning_output = object_for_data_set.get_data_in_range_from_the_data_set(file_path,1,1)
object_for_data_set.visualize_one_variable_data(traning_input,traning_output)
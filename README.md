# Linear-Regression-for-predictions
This repository contains two dataset 

ex1data1.txt
=>
The First column is the population of a city and the second column is
the proot of a food truck in that city. A negative value for prot indicates a
loss.

ex1data2.txt
=> The First column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

Flow is as follows:
1) Read the data form the data set and devide the data in feature data(traning input) and output data(traning output).
2) Set number of iterations and learning rate.
3) For each iteration we calculate the gradient and follow the direction where the overall cost function will be mininmum.
   * To test if the implementationn is correct or not always check after every iteration the cost function should reduce. If in case it is increasing then might be the implementaion of gradient descent wrong. 
   * The selection of learning rate plays an important role, i.e. if the learning rate is small the convergence is slow, in case the learning rate is high then the function converges and might lose the local minimum as well and the function start increasing again.
   
compute_cost_function_for_data(X,theta,Y)
=> It computes the value of cost function at any given point with the learned parameters. This function is used in the gradient algorith which gives us a proof whether the function is minimizing or not with each iteration

gradient_descent_for_function_minimization(number_of_iteration,learning_rate,X,theta,Y)
=> This is the actual gradient descent algorithm, which is based on the concept of gradient. We know that a gradient is a overall derivative of a multivariable function which at a given point always directed to the steepest ascent. So to mininimize the function we have to go into the opposite direction of the gradient.  

The hypothesis functions for 
1) Single variable:
     h(theta) = theta_0 + theta_1 * x_1
2) Multiple Variable:
     h(theta) = theta_0 + theta_1 * x_1 + theta_2 * x_2
     
     We have not used any non-linear function here only linear function only.

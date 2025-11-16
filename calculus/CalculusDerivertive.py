import sympy as sp
import numpy as np
# x = sp.symbols('x')
# f = x**2
# derivative = sp.diff(f,x)
# print(derivative )
# x,y = sp.symbols('x y')
# f =x**2 + y**2 + 4*x*y
# grad_x =sp.diff(f,x)
# grad_y =sp.diff(f,y)

# print(grad_x,"\n",grad_y)

# def gradient_descent(X,y,theta,learning_rate,iterations):
#     m = len(y)
#     for _ in range(iterations):
#         predictions = np.dot(X,theta)
#         errors = predictions - y
#         gradient= (1/m)*np.dot(X.T,errors)
#         theta -= learning_rate * gradient
#     return theta

# X = np.array([[1,1],[1,2],[1,3]])
# y = np.array([2,2.5,3.5])
# theta = np.array([0.1,0.1])
# learning_rate = 0.1
# iterations = 1000000

# optimized_theta = gradient_descent(X,y,theta,learning_rate,iterations)

# print(optimized_theta)
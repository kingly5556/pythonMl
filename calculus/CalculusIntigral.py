import sympy as sp
import numpy as np
# x = sp.symbols('x')
# f = x**2
# definite_integral = sp.integrate(f,(x,2,0))
# indefinite_integral = sp.integrate(f,x)
# print(definite_integral,"\n",indefinite_integral)

np.random.seed(42)
X = 2*np.random.rand(100,1)
y = 4+3*X+np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)),X]

def stochastic_gradient_descent(X,y,theta,learning_rate,n_epochs):
    m = len(y)
    for epoch in range(n_epochs):
        for _ in range(m):
            random_index = np.random.randint(m)
            xi= X[random_index:random_index+1]
            yi=y[random_index:random_index+1]
            gradients =2*xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradients
    return theta

theta = np.random.rand(2,1)
learning_rate = 0.01
n_epochs = 500

theta_opt = stochastic_gradient_descent(X_b,y,theta,learning_rate,n_epochs)
print("Optimized Parameters: ",theta_opt)
import numpy as np 

def f(x, v, omega, dt): 
   x, y, theta = x


x_new = x + v * np.cos(theta) * dt
y_new = y + v * np.sin(theta) * dt

theta_new = theta + omega * dt

return np.array([x_new, y_new, theta_new])
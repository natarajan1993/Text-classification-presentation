import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-20,20)

y = 1/(1+np.exp(-x))

plt.plot(x,y)

plt.show()
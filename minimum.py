import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,5,0.001)
y = x**4 - 3*x**3 + 2

plt.plot(x,y)
plt.show()

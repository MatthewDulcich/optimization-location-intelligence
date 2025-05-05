import matplotlib.pyplot as plt
import numpy as np


func = lambda P: P**1.5 * np.exp(-P**2/10) / (P + 1000) 

x = np.linspace(0, 100, 1000)
y = func(x)
plt.plot(x, y, label='Function')
plt.xlabel('P')
plt.ylabel('f(P)')
plt.title('Function Plot')
plt.legend()
plt.grid(True)
plt.show()
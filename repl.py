# %% Cell 1
# %% Cell 2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

# %%
# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8,6))
plt.plot(x, y, 'b-', label='sin(x)')
plt.title('Simple Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

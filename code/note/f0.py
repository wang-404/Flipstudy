import matplotlib .pyplot as plt
import numpy as np
from bokeh.plotting import output_notebook,show
output_notebook(hide_banner=True)
import plotly
plotly.offline.init_notebook_mode()
x = np.linspace(-2 * np.pi,2*np.pi,100)
y = np.sin(x/2)
z = np.cos(x/4)
plt.title("Matplotlib Figure in Bokeh")
plt.plot(x, y, "r-", marker='o')
plt.plot(x, z, "g-x", linestyle="-.")
pfig = plotly.tools.mpl_to_plotly(plt.gcf())


import numpy as np
import matplotlib.pyplot as plt
from skimage import data

colorwheel = data.colorwheel()
plt.imshow(colorwheel)

imagemParaMatriz = np.asarray(colorwheel)
print(imagemParaMatriz.shape)
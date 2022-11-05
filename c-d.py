import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from numpy.linalg import svd

colorwheel = data.colorwheel()
plt.imshow(colorwheel)
# convertendo pra p&b
grayscale_colorwheel = rgb2gray(colorwheel)

# calculando SVD
U,S,V_T = svd(grayscale_colorwheel, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(3, 2, figsize=(8, 20))

curr_fig=0
for r in [60, 70, 80]:
  colorwheel_approx=U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
  ax[curr_fig][0].imshow(256-colorwheel_approx)
  ax[curr_fig][0].set_title("k = "+str(r))
  ax[curr_fig,0].axis('off')
  ax[curr_fig][1].set_title("Imagem de referência")
  ax[curr_fig][1].imshow(grayscale_colorwheel)
  ax[curr_fig,1].axis('off')
  curr_fig += 1
plt.show()
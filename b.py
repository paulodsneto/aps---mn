import numpy as np
from numpy.linalg import svd

X = np.array([[2, 1, 2], 
              [-2,3,-1]])
print("-------------------------------")
print("Matriz original: ")
print(X)
print("-------------------------------")
#SVD
U, Singular, V = svd(X)
print("-------------------------------")
print("U: ",U)
print("-------------------------------")
print("Valores Singulares:", Singular)
print("-------------------------------")
print("V^{T}",V.T)
print("-------------------------------")
k=1
rr = np.dot(U[:, :k], np.dot(np.diag(Singular[:k]), V[:k, :]))
print("-------------------------------")
print("rr",rr)
print("-------------------------------")
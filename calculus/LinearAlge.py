import numpy as np
A = np.array([[2,3],[1,4]])
# determinant = np.linalg.det(A)

inverse = np.linalg.inv(A)
print(inverse,"\n")

u,s,vt = np.linalg.svd(A)
print("\n",u)
print("\n",s)
print("\n",vt)

eigvals, eigvec = np.linalg.eig(A)

print("\n",eigvals)
print("\n",eigvec)
import numpy as np

a23 = np.array([[1, 1, 1], [2, 2, 2]])
a332 = np.array([[[1, 2], [3, 4], [5, 6]], [[1, 1], [1, 2], [2, 2]], [[1, 2], [1, 3], [1, 4]]])

print(a23[1, :])
print(a332[0, :, 0])
print(sum(a23[1, :] * a332[0, :, 0]))

x1 = np.matmul(a23, a332)
x2 = np.dot(a23, a332)
print(a23.shape)
print(a332.shape)
print(x1.shape)
print(x2.shape)

x3 = np.matmul(a332, a23)
print(x3.shape)

np.random.seed(42)
np.set_printoptions(precision=3)

a223 = np.random.random(size=(2, 2, 3))
a234 = np.random.random(size=(2, 3, 4))

x1 = np.matmul(a223, a234)
x2 = np.dot(a223, a234)
print(x1.shape)
print(x2.shape)

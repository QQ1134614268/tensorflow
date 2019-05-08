import numpy as np
sizes = [3, 2, 1]
for x, y in zip(sizes[:], sizes[1:]):
    print(x, y)
#     print(np.random.randn(y, x))
    
print(np.random.randn(2, 2,3))


a = [1, 2, 3]
b = [4, 5, 6]
c = [4, 5, 6, 7, 8]
zipped = zip(a, b)  # 打包为元组的列表
zip(a, c)  # 元素个数与最短的列表一致
zip(*zipped)  # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式

for x, y in zip(a, c):
    print(x, y)

a=np.array([[1,2],[3,4]])
b=np.array([[1,2],[3,4]])
print(a)
print(a.dot(b))
print(a*b)

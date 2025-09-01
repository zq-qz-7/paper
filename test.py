from copy import deepcopy

a =[1, 2, 3, 4]
b = deepcopy(a)
b[0] = 5
print(a)
print(b)
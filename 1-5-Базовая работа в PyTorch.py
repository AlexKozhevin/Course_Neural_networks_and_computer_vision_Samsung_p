import torch

"""
Реализуйте при помощи pyTorch функцию, которая возвращает сумму (x.sum()) элементов тензора X, 
строго превышающих значение limit, которое является входным значением алгоритма.
Входная матрица: X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
"""

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())
mask = X > limit
larger_than_limit_sum = X[mask].sum()
print(larger_than_limit_sum)

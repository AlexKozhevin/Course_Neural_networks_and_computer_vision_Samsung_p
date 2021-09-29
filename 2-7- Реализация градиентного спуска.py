import torch

"""
Реализуйте расчет градиента для функции:
f(w)=i,j∏loge(loge(wi,j+7)) в точке w = [[5, 10], [1, 2]]w =[[5,10],[1,2]]
"""

w = torch.tensor(
    [[5.,  10.],
     [1.,  2.]], requires_grad=True)
function =  torch.log(torch.log(w + 7)).prod()
function.backward()
print(w.grad)  # Код для самопроверки

"""
Реализуйте градиентный спуск для той же функции.
Пусть начальным приближением будет w^{t=0} = [[5, 10], [1, 2]], шаг градиентного спуска alpha=0.001.
Чему будет равен w^{t=500}?
"""

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001

for _ in range(500):
    # it's critical to calculate function inside the loop:
    function = (w + 7).log().log().prod()
    function.backward()
    w.data -= alpha * w.grad
    w.grad.zero_()
print(w)  # Код для самопроверки


"""
Перепишите пример, используя torch.optim.SGD
"""

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001
optimizer = torch.optim.SGD([w], lr=0.001)

for _ in range(500):
    # it's critical to calculate function inside the loop:
    function = (w + 7).log().log().prod()
    function.backward()
    optimizer.step()
    w.grad.zero_()
print(w)  # Код для самопроверки

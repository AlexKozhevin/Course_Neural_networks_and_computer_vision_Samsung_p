import torch
import datetime

"""
Обучим нейронную сеть для задачи регрессии:
Возьмем более сложную функцию в качестве таргета: y=2^x sin(2^{-x})
Получите метрику не хуже 0.03
Что можно варьировать:
1) Архитектуру сети
2) loss-функцию
3) lr оптимизатора
4) Количество эпох в обучении
"""


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        in_features = 1
        out_features = n_hidden_neurons
        self.fc1 = torch.nn.Linear(in_features, out_features, bias=True)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(out_features, out_features, bias=True)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(out_features, in_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


def target_function(x):
    return 2**x * torch.sin(2**-x)


def loss(pred, target):
    MAE = abs(pred - target)
    return MAE.mean()


def metric(pred, target):
    return (pred - target).abs().mean()


start_time = datetime.datetime.now()
net = RegressionNet(50)

# ------Dataset preparation start--------:
x_train =  torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch_index in range(200):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

# Проверка осуществляется вызовом кода:
print(metric(net.forward(x_validation), y_validation).item())
print('Time elapsed:', datetime.datetime.now() - start_time)

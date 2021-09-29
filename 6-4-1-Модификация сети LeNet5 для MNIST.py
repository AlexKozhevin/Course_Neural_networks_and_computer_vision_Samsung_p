import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt

"""
    Давайте видеоизменим LeNet чтобы повысить качество на валидации. У нас есть некоторые вещи которые неплохо было бы 
исправить. 
    Например, активации: сейчас уже никто не использует активации тангенсами, потому что они приводят к 
затуханию градиента -- вы не можете построить действительно глубокую сеть, чтобы ваши ошибки, посчитанные и в конце 
сети хорошо прокидывались к началу сети. Тангенсы или сигмоиды приводят к тому, что у вас сигнал очень быстро затухает. 
    Во-вторых сейчас мало кто использует свёртки 5 на 5. Практически все сети свёртки 5 на 5 заменяют на подряд идущие 
две свёртки 3 на 3. Почему так делается? Потому что в 5 на 5 -- 25 весов, как можно легко догадаться, а вот в двух 
свёртках 3 на 3 -- в одной свёртке 9 весов, и в другой 9 весов -- получается 18 весов. А, в принципе, тот объём 
многообразия, которое позволяет применение двух последовательных сверток, аналогичен одной свёртке 5 на 5. 
Весов меньше -- наверное, будет меньше переобучения. 
    В-третьих у нас тут используется average pooling, но average pooling уже нигде не 
используется, кроме как в самом конце сети. Сейчас используются везде max pooling. Давайте это тоже поправим. 
И последний момент -- это батч-нормализация, которая призвана ускорять обучение. Кажется, тут её тоже логично применить.
    Давайте все эти пункты мы возьмём и попробуем. У нас появились некоторые переменные: 
activation, pooling, conv_size и use_batch_norm, которые определяют -- активация будет тангенсом 
или ReLU, pooling будет "average" или "max pooling". "conv_size = 5" -- это значит, будет одна свёртка 5 на 5, либо, 
если мы поставим сюда значение "3", то будет две последовательных 3 на 3. И переменная use_batch_norm определяет, будет
ли использована батч-нормализация, или нет. А функция forward теперь выглядит следующим образом. 
Если у нас "conv_size = 5", то мы используем одну конволюцию 5 на 5; если у нас "conv_size=3", 
то мы будем использовать сначала одну конволюцию 3 на 3, и результат её передадим в следующую конволюцию 3 на 3. 
    Далее у нас будет активация, соответственно -- в инициализации мы сказали: act1 - это будет либо ReLU, либо тангенс.
И если мы хотим батч-нормализацию, мы можем дополнительно, после свёртки применять слой батч-нормализации batchnorm1 
или batchnorm2. 
    Надо отдельно сказать про слои батч-нормализации. Они вызываются с помощью слоя torch.nn.BatchNorm2d, 
потому что мы имеем дело с картинками. Если бы мы хотели нормализовать некоторый вектор после, например fully-connected 
слоя, то мы бы использовали torch.nn.BatchNorm1d. И на вход нужно передать num_features, то есть то количество каналов, 
которое имеет картинка или тензор перед батч-нормализацией.
"""


class LeNet5(torch.nn.Module):
    def __init__(self,
                 activation='tanh',
                 pooling='avg',
                 conv_size=5,
                 use_batch_norm=False):
        super(LeNet5, self).__init__()

        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm

        if activation == 'tanh':
            activation_function = torch.nn.Tanh()
        elif activation == 'relu':
            activation_function  = torch.nn.ReLU()
        else:
            raise NotImplementedError

        if pooling == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'max':
            pooling_layer  = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        if conv_size == 5:
            self.conv1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, padding=2)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(
                in_channels=6, out_channels=6, kernel_size=3, padding=1)
        else:
            raise NotImplementedError

        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = pooling_layer

        if conv_size == 5:
            self.conv2 = torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, padding=0)
        elif conv_size == 3:
            self.conv2_1 = torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=3, padding=0)
            self.conv2_2 = torch.nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, padding=0)
        else:
            raise NotImplementedError

        self.act2 = activation_function
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = pooling_layer

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = activation_function

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = activation_function

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        if self.conv_size == 5:
            x = self.conv1(x)
        elif self.conv_size == 3:
            x = self.conv1_2(self.conv1_1(x))
        x = self.act1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.pool1(x)

        if self.conv_size == 5:
            x = self.conv2(x)
        elif self.conv_size == 3:
            x = self.conv2_2(self.conv2_1(x))
        x = self.act2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


def trains(net, X_train, y_train, X_test, y_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = 100

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(30):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index+batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        net.eval()
        test_preds = net.forward(X_test)
        test_loss_history.append(loss(test_preds, y_test).data.cpu())

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
        test_accuracy_history.append(accuracy)

        print(accuracy)
    print('---------------')
    return test_accuracy_history, test_loss_history


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.data
y_train = MNIST_train.targets
X_test = MNIST_test.data
y_test = MNIST_test.targets
len(y_train), len(y_test)

plt.imshow(X_train[0, :, :])
plt.show()
print(y_train[0])

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()
print(X_train.shape)

accuracies = {}
losses = {'tanh': (trains(LeNet5(activation='tanh', conv_size=5),
                          X_train, y_train, X_test, y_test))[1], 'relu': (trains(LeNet5(activation='relu', conv_size=5),
                                                                                 X_train, y_train, X_test, y_test))[1],
          'relu_3': (trains(LeNet5(activation='relu', conv_size=3),
                            X_train, y_train, X_test, y_test))[1],
          'relu_3_max_pool': (trains(LeNet5(activation='relu', conv_size=3, pooling='max'),
                                     X_train, y_train, X_test, y_test))[1],
          'relu_3_max_pool_bn': (trains(LeNet5(activation='relu', conv_size=3, pooling='max', use_batch_norm=True),
                                        X_train, y_train, X_test, y_test))[1]}

for experiment_id in accuracies.keys():
    plt.plot(accuracies[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Accuracy')
plt.show()

for experiment_id in losses.keys():
    plt.plot(losses[experiment_id], label=experiment_id)
plt.legend()
plt.title('Validation Loss')
plt.show()

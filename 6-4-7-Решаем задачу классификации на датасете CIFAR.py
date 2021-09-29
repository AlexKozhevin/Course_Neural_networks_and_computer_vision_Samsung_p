import random
import numpy as np
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets
import matplotlib.pyplot as plt

"""
    На прошлом шаге мы пробовали конволюционную сеть на датасете MNIST, Дело в том, что даже классические методы на этом 
датасете показывают довольно хорошие результаты (в районе 98 -- 99%). 
    Давайте возьмем действительно сложный датасет, с которым классические методы не справляются. 
Есть такой датасет: CIFAR-10, он состоит из картинок 32 на 32, это RGB-картинки, всего картинок у них 60 тысяч - 
50 тысяч в трейне и 10 тысяч в валидации. Есть ещё и его увеличенная версия -- CIFAR-100, там 100 классов, но мы, для 
простоты, возьмём CIFAR-10. Обучим на нём наши сети, которые были в предыдущем шаге. Наш код практически не изменится. 
    Мы должны немного изменить загрузку датасета, потому что теперь у нас три канала в изображении. А во-вторых у нас 
изменится сеть, потому что раньше нам приходили изображения 28 на 28, теперь 32 на 32. Давайте загрузим этот датасет
и посмотрим на него. Загрузить этот датасет, точно так же, как MNIST, мы можем с помощью библиотеки torchvision. 
    Cформируем CIFAR_train и CIFAR_test, Нам нужно преобразовать эти датасеты в FloatTensor, если мы говорим о 
картинках, и в LongTensor, если мы говорим о классах. Видим, что, действительно, разбивка совпадает с заявленной: 
50 000 идёт в train, 10 000 идёт в test. Если мы посмотрим на максимальное и минимальное значения в картинках, 
то окажется, что минимальное значение -- "0", а максимальное -- "255". Действительно, у них каждый пиксель кодируется 
значениями от 0 до 255 для каждого канала. С такими изображениями можно работать -- есть сети, которые действительно 
работают с такими данными. Но мы, для удобства, отнормируем эти данные -- разделим каждый пиксель, каждое его значение, 
на 255 и получим, что в наших картинках будут лежать значения от нуля до единицы. 
    Давайте так и сделаем, разделим на 255. Кроме того, мы можем посмотреть на классы (действительно, 10 классов), и 
визуализировать. Действительно видим -- картинки картинки 32 на 32, тут изображены 10 классов, метки -- сверху. 
Например, метка 9 отвечает за "truck", то есть за грузовики, то есть у нас разметка правильная. Ещё одна особенность 
этого датасета: как и у обычных картинок, канал "цвет" кодируется в последней размерности. То есть, сначала идёт высота 
картинки, ширина, а после этого уже цвет. Но pytorch требует, чтобы этот канал шёл на первом месте. То есть, мы имеем 
сейчас 4-мерный тензор: "количество картинок, высота, ширина и цвет", а Pytorch хочет: "количество картинок, количество 
цветов, ширина, высота". И теперь нам нужно реорганизовать размерность нашего тензора таким образом, чтобы цвет шёл на 
втором месте -- как раз после количества картинок в датасете. Это делается с помощью метода "permute". Вот здесь у нас 
permute с четырьмя аргументами. Первый аргумент "0" отвечает за количество картинок в нашем датасете. Мы не хотим, 
чтобы "количество картинок" в датасете поменяло позицию, соответственно тут стоит ноль. Далее стоит число "3" -- это 
значит, что на это место придёт размерность, которая была под номером "3" в изначальном тензоре, то есть это будет 
"количество каналов". Далее идёт "1", то есть на это место придёт "высота картинки", а дальше "2" -- это значит -- 
на это место придёт "ширина картинки", и у нас, после выполнения этой операции, у всего датасета будет shape: 50 000 
на 3 на 32 на 32. то есть "каналы" будут идти перед размерностью изображения. 
    Далее -- мы изменим нашу сеть LeNet такимобразом, чтобы она принимала изображения 32 на 32 и три канала на входе. 
Чтобы передать ей три канала, нужно в in_chanels первой же конволюции поставить "3" (раньше у нас тут было "1"). 
А вот размерность 32 на 32 у нас получается из паддинга. Как вы помните, в оригинальном LeNet была размерность 32 на 32 
и там были нулевые паддинги, то есть конволюции не выходили за изображение. Мы специально, в LeNet, для MNIST 
это изменяли, чтобы у нас первая конволюция выходила за изображение и после первой конволюции получается разрешение 
28 на 28. А теперь мы, наоборот, хотим чтобы паддинга не было, и из размерности 32 на 32 получалась размерность 28 на 28
Соответственно, мы поставим паддинг "0". 
    Больше ничего не меняется -- процесс обучения не изменяется, он остался таким же, как и раньше.
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
                in_channels=3, out_channels=6, kernel_size=5, padding=0)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=3, padding=0)
            self.conv1_2 = torch.nn.Conv2d(
                in_channels=6, out_channels=6, kernel_size=3, padding=0)
        else:
            raise NotImplementedError

        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = pooling_layer

        if conv_size == 5:
            self.conv2 = self.conv2 = torch.nn.Conv2d(
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


"""
На графике validation accuracy мы видим, что наша новая сеть, которую мы назвали CIFARNet, побеждает все предыдущие 
реализации. И, видимо, это происходит из-за того, что у неё больше фильтров в конволюциях. Давайте посмотрим ещё графики
лоссов, и на этом графике видно, что после 3-й эпохи обучать сеть уже бессмысленно, она не улучшается по качеству, 
но её уверенности становятся более категоричными. То есть сеть становится более уверена даже в своих неправильных 
предсказаниях, из-за чего график лосса сильно растёт.
"""


class CIFARNet(torch.nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.act1 = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = torch.nn.ReLU()
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = torch.nn.ReLU()
        self.batch_norm3 = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(8 * 8 * 64, 256)
        self.act4 = torch.nn.Tanh()
        self.batch_norm4 = torch.nn.BatchNorm1d(256)

        self.fc2 = torch.nn.Linear(256, 64)
        self.act5 = torch.nn.Tanh()
        self.batch_norm5 = torch.nn.BatchNorm1d(64)

        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.batch_norm3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act4(x)
        x = self.batch_norm4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.batch_norm5(x)
        x = self.fc3(x)

        return x


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def train(net, X_train, y_train, X_test, y_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = 50

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(10):
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

            # print(X_batch)

        net.eval()

        # test_preds = net.forward(X_test)
        # test_loss_history.append(loss(test_preds, y_test).data.cpu())
        #
        # accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
        # test_accuracy_history.append(accuracy)

        with torch.no_grad():
            test_preds = net.forward(X_test)
            loss_value = loss(test_preds, y_test).item()
            test_loss_history.append(loss_value)

            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().item()
            test_accuracy_history.append(accuracy)

        print(accuracy)
    del net
    return test_accuracy_history, test_loss_history


"""
    Итак, мы успели поупражняться с построением собственных архитектур, однако сейчас редко кто трудится над 
конструированием новой архитектуры для популярных задач CV. Победитель конкурса классификации на ImageNet каждый год 
меняется, тем не менее, самый популярный в бытовом использовании вариант – это ResNet.
Модификаций данной сети довольно много: ResNet13, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, которые были 
созданы для ImageNet. А так же есть ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202 для датасета CIFAR10.
    Предлагаем проверить эти сетки на прочность!
Из библиотеки torchvision (ставится вместе с pytorch), можно проимпортировать ResNet18 командой: 
    "from torchvision.models import resnet18"
Вот так просто. Сравните результаты resnet18 и CIFARNet. 
    Какая сеть дает лучший результат?
    Попробуйте, пользуясь нашей лекцией и описанием архитектуры из оригинальной статьи, написать собственную реализацию 
ResNet20. Если возникнут сомнения, можно свериться с кодом из https://github.com/akamaster/pytorch_resnet_cifar10 . 
    Удалось ли побить resnet18?
    Реализуйте ResNet110 (возможно, придется уменьшить размер batch'a). Проверьте утверждение, что ResNet110 не 
обучается (или обучается в 10% случаев), если отключить BatchNorm.
    Добавьте Dropout2d после каждого BatchNorm2d для ResNet20. Есть ли положительный эффект? Как параметр "p" этого слоя 
влияет на accuracy и на переобучение? 
    Добавьте l2-регуляризацию. В PyTorch она активируется с помощью параметра weight_decay в оптимизаторе. Значение обычно 
выбирают из [1e-3, 1e-4, 1e-5]. Значение 1e-2 ставить не стоит, т.к. сеть не сможет учиться,  а 1e-6 скорее всего 
просто не повлияет на обучение (но лучше это проверить это утверждение самостоятельно). Пример:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
"""

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

CIFAR_train = torchvision.datasets.CIFAR10('./', download=True, train=True)
CIFAR_test = torchvision.datasets.CIFAR10('./', download=True, train=False)

X_train = torch.FloatTensor(CIFAR_train.data)
y_train = torch.LongTensor(CIFAR_train.targets)
X_test = torch.FloatTensor(CIFAR_test.data)
y_test = torch.LongTensor(CIFAR_test.targets)

len(y_train), len(y_test)

X_train.min(), X_train.max()
print('Classes: ', CIFAR_train.classes)

plt.figure(figsize=(20, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].int())
plt.show()

print(X_train.shape, y_train.shape)

X_train = X_train.permute(0, 3, 1, 2)
X_test = X_test.permute(0, 3, 1, 2)

print(X_train.shape)

accuracies = {}
losses = {}

print('ALL_NETs_____________________________________________________________________________________________________')
print()
print('LeNet5_tanh__________________________________________________________________________________________________')
accuracies['tanh'], losses['tanh'] = \
    train(LeNet5(activation='tanh', conv_size=5),
          X_train, y_train, X_test, y_test)
print()
print('LeNet5_relu__________________________________________________________________________________________________')
accuracies['relu'], losses['relu'] = \
    train(LeNet5(activation='relu', conv_size=5),
          X_train, y_train, X_test, y_test)
print()
print('LeNet5_relu_3________________________________________________________________________________________________')
accuracies['relu_3'], losses['relu_3'] = \
    train(LeNet5(activation='relu', conv_size=3),
          X_train, y_train, X_test, y_test)
print()
print('LeNet5_relu_3_max_pool_______________________________________________________________________________________')
accuracies['relu_3_max_pool'], losses['relu_3_max_pool'] = \
    train(LeNet5(activation='relu', conv_size=3, pooling='max'),
          X_train, y_train, X_test, y_test)
print()
print('LeNet5_relu_3_max_pool_bn____________________________________________________________________________________')
accuracies['relu_3_max_pool_bn'], losses['relu_3_max_pool_bn'] = \
    train(LeNet5(activation='relu', conv_size=3, pooling='max', use_batch_norm=True),
          X_train, y_train, X_test, y_test)
print()
print('CiFAR_net____________________________________________________________________________________________________')
accuracies['cifar_net'], losses['cifar_net'] = \
        train(CIFARNet(), X_train, y_train, X_test, y_test)
print()
print('resnet18_____________________________________________________________________________________________________')
accuracies['resnet18'], losses['resnet18'] = \
    train(resnet18(), X_train, y_train, X_test, y_test)
print()
print('resnet20_____________________________________________________________________________________________________')
accuracies['resnet20'], losses['resnet20'] = \
    train(resnet20(), X_train, y_train, X_test, y_test)
print()
print('resnet50_______________________________________________________________________________________________________')
accuracies['resnet50'], losses['resnet50'] = \
    train(resnet20(), X_train, y_train, X_test, y_test)

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

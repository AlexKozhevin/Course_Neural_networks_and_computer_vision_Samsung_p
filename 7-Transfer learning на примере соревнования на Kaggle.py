import os
import pandas as pd
import zipfile
import shutil
from tqdm import tqdm
import torch
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
from torch.utils.data import dataloader
from torchvision import models
import random

"""
Предлагаем поучаствовать в соревновании на Kaggle "Dirty vs Cleaned V2"
https://www.kaggle.com/c/platesv2
Чтобы получить за него баллы, отправьте свой submission.csv в форму этого задания. Мы начисляем от 0 до 20 баллов,
в зависимости от accuracy: 80% даст 0 баллов, 100% - 20 баллов.
"""


class ImageFolderWithPaths(tv.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def show_input(input_tensor, title=''):
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)


def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # У каждой эпохи есть этап обучения и валидации.
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                # scheduler.step()
                model.train()  # Установить модель в режим обучения

            else:
                dataloader = val_dataloader
                model.eval()   # Установить модель в режим валидации

            running_loss = 0.
            running_acc = 0.

            # Перебираем данные.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + оптимизировать, только если в фазе обучения
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # статистика
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model


if __name__ == "__main__":
    seed = 51
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    print(os.listdir("input"))

    with zipfile.ZipFile('input/plates.zip', 'r') as zip_obj:
        # Извлечь все содержимое zip-файла в текущий каталог
        zip_obj.extractall('kaggle/working/')

    print('After zip extraction:')
    print(os.listdir("kaggle/working/"))

    data_root = 'kaggle/working/plates/'
    print(os.listdir(data_root))

    train_dir = 'train'
    val_dir = 'val'
    class_names = ['cleaned', 'dirty']

    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(data_root, 'train', class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % 6 != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomApply([
            tv.transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4
            )
        ]),
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = tv.datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = tv.datasets.ImageFolder(val_dir, val_transforms)

    batch_size = 8
    train_dataloader = dataloader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = dataloader.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print(len(train_dataloader), len(train_dataset))

    X_batch, y_batch = next(iter(train_dataloader))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean)
    plt.show()

    for x_item, y_item in zip(X_batch, y_batch):
        show_input(x_item, title=class_names[y_item])

    # выбираем нейросеть
    # model = models.resnet50(pretrained=True)
    model = models.resnet152(pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet101_2', pretrained=True)

    # Замораживаем сеть, т.е. не обучаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # заменяем последний полносвязный слой с выходом два класса (нейрона)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Обучаем
    train_model(model, loss, optimizer, scheduler, num_epochs=5)

    test_dir = 'test'
    shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
    test_dataset = ImageFolderWithPaths('test', val_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print(test_dataset)

    # Валидация
    model.eval()

    test_predictions = []
    test_img_paths = []
    for inputs, labels, paths in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
        test_img_paths.extend(paths)

    test_predictions = np.concatenate(test_predictions)
    inputs, labels, paths = next(iter(test_dataloader))

    for img, pred in zip(inputs, test_predictions):
        show_input(img, title=pred)
        plt.show()

    # Формируем 'submission.csv'
    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '', regex=True)
    submission_df['id'] = submission_df['id'].str[-4:]
    submission_df.set_index('id', inplace=True)
    submission_df.head(n=6)
    submission_df.to_csv('submission.csv')

    shutil.rmtree('train')
    shutil.rmtree('val')
    shutil.rmtree('test')
    shutil.rmtree('kaggle')

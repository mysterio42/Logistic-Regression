import random
import string

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from utils.data import test_ds
from utils.plot import display_digit


def load_model(model, path_extension: str):
    model.load_state_dict(torch.load(path_extension))


def generate_model_name(size=5):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(model, train_ds_loader, test_ds_loader, learning_rate, n_epochs, save_model):
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    it = 0
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_ds_loader):

            images = images.view(-1, 28 * 28)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            it += 1

            if it % 500 == 0:
                correct = 0
                total = 0
                test_model = model.train(False)
                for test_images, test_labels in test_ds_loader:
                    test_images = test_images.view(-1, 28 * 28)

                    test_logits = test_model(test_images)

                    _, predicted = torch.max(test_logits, 1)

                    total += test_labels.size(0)

                    correct += (predicted == test_labels).sum()

                accuracy = 100 * int(correct) / total
                print('Iteration: {} Loss: {} Accuracy {}'.format(it, loss.item(), accuracy))
    if save_model:
        torch.save(model.state_dict(), 'weights/model-' + generate_model_name(5) + '.pkl')


def predict_model(model):
    idx = random.choice(range(len(test_ds)))
    test_image = test_ds[idx][0]
    label_real = test_ds[idx][1]

    logits = model(test_image.reshape(-1, 28 * 28))
    _, label_predicted = torch.max(logits, 1)

    display_digit(test_image, label_real, label_predicted)

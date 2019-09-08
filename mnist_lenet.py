import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.attacks import GradientSignAttack
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH


seed = 0
mode = 'adv' # 'cln' or 'half' or 'adv'
train_bs = 50 # train batch size
test_bs = 1000 # test batch size
log_interval = 200


torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if mode == 'cln':
    nb_epoch = 10
    model_filename = 'mnist_lenet5_clntrained.pth'
elif mode == 'half':
    nb_epoch = 50
    model_filename = "mnist_lenet5_halftrained.pth"
elif mode == 'adv':
    nb_epoch = 50
    model_filename = "mnist_lenet5_advtrained.pth"
else:
    raise RuntimeError('mode must be "cls" or "adv"')


train_loader = get_mnist_train_loader(
    batch_size=train_bs, shuffle=True)
test_loader = get_mnist_test_loader(
    batch_size=test_bs, shuffle=False)

model = LeNet5()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


adversary = GradientSignAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=0.3, clip_min=0.0, clip_max=1.0, targeted=False)


for epoch in range(nb_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ori = data
        if mode == 'adv' or (mode == 'half' and random.random() > 0.5):
            # when performing attack, the model needs to be in eval mode
            # also the parameters should be accumulating gradients
            with ctx_noparamgrad_and_eval(model):
                data = adversary.perturb(data, target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(
            output, target, reduction='elementwise_mean')
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    model.eval()
    test_clnloss = 0
    clncorrect = 0

    test_advloss = 0
    advcorrect = 0

    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        advdata = adversary.perturb(clndata, target)
        with torch.no_grad():
            output = model(advdata)
        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(target.view_as(pred)).sum().item()

    test_clnloss /= len(test_loader.dataset)
    print('\nTest set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)\n'.format(
              test_clnloss, clncorrect, len(test_loader.dataset),
              100. * clncorrect / len(test_loader.dataset)))
    
    test_advloss /= len(test_loader.dataset)
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)\n'.format(
              test_advloss, advcorrect, len(test_loader.dataset),
              100. * advcorrect / len(test_loader.dataset)))


torch.save(
    model.state_dict(),
    os.path.join(TRAINED_MODEL_PATH, model_filename))

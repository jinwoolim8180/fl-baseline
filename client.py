import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def zero_grad(model):
    grad = {k: torch.zeros(v.shape).to(v.device) for k, v in model.state_dict().items()}
    return grad


def local_train(args, setting, lr, c, model, dataloader, device):
    # hyperparameter setting
    K = 0
    alpha = setting.alpha
    if setting.c_i is None:
        c_i = zero_grad(model)
    else:
        c_i = setting.c_i
    if c is None:
        c = zero_grad(model)
    for k in c.keys():
        c[k] = c[k].to(device)

    # optimizer and scheduler setting
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    # copy previous parameter
    epoch_loss = 0
    prev_param = copy.deepcopy(model.state_dict())
    for e in range(args.local_ep):
        model.train()
        batch_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            log_probs = model(images)
            loss = F.cross_entropy(log_probs, labels)
            if args.fed_strategy == 'feddyn':
                loss += 0.5 * alpha * (model.parameters() - prev_param).norm(2)**2 - model.parameters() * c_i
            elif args.fed_strategy == 'scaffold':
                loss += model.parameters() * (c - c_i)
            loss.backward()
            optimizer.step()
            scheduler.step()
            K += 1
            batch_loss += loss.item()
        epoch_loss += batch_loss
    epoch_loss /= K

    # hyperparamter changing
    if args.fed_strategy == 'scaffold':
        setting.c_i -= (model.parameters() - prev_param) / (K * lr) + c
    elif args.fed_strategy == 'feddyn':
        setting.c_i -= alpha * (model.parameters() - prev_param)
        setting.alpha = 1 / (K * lr)

    return model.state_dict(), epoch_loss

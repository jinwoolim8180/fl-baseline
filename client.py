import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_grad(model):
    grad = {k: torch.zeros(v.shape).to(v.device) for k, v in model.state_dict().items()}
    return grad


def _grad_norm(net):
    norm = 0
    for _, param in net.named_parameters():
        norm += param.grad.data.double().norm(2).item()
    return math.sqrt(norm)


def dict_to_device(dict, device):
    for k in dict.keys():
        dict[k] = dict[k].detach().to(device)


def local_train(args, lr, c_i, c, model, dataloader, device):
    # hyperparameter setting
    K = 0
    if c_i is None:
        c_i = zero_grad(model)
    if c is None:
        c = zero_grad(model)
    dict_to_device(c_i, device)
    dict_to_device(c, device)

    # optimizer and scheduler setting
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    # copy previous parameter
    epoch_loss = 0
    prev_param = copy.deepcopy(model.state_dict())

    # training
    for e in range(args.local_ep):
        model.train()
        batch_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = F.cross_entropy(log_probs, labels)
            if args.fed_strategy == 'feddyn':
                for k, v in model.named_parameters():
                    loss += 0.5 * args.alpha * (v - prev_param[k]).norm(2)**2 - torch.sum(v * c_i[k])
            elif args.fed_strategy == 'scaffold':
                for k, v in model.named_parameters():
                    loss += torch.sum(v * (c[k] - c_i[k]))
            loss.backward()
            if args.fed_strategy == 'implicit':
                    param_copy = copy.deepcopy(model.state_dict())
                    grad_copy = {k: copy.deepcopy(v.grad.data).to(v.grad.device) for k, v in model.named_parameters()}
                    norm = _grad_norm(model) + 1e-7
                    for name, param in model.named_parameters():
                        param.data += 0.1 / norm * param.grad.data

                    log_probs = model(images)
                    loss = F.cross_entropy(log_probs, labels)
                    loss.backward()
                    for name, param in model.named_parameters():
                        param.grad.data *= args.lamda * norm / 0.1
                        param.grad.data += (1 - args.lamda * norm / 0.1) * grad_copy[name]
                    model.load_state_dict(param_copy)
            elif args.fed_strategy == 'feddyn':
                for name, param in model.named_parameters():
                    param.grad.data += args.alpha * (param.data - prev_param[name]) - c_i[name]
            optimizer.step()
            K += 1
            batch_loss += loss.item()
        scheduler.step()
        epoch_loss += batch_loss
    epoch_loss /= K

    # hyperparamter changing
    if args.fed_strategy == 'scaffold':
        for k, v in model.named_parameters():
            c_i[k] -= (v - prev_param[k]) / (K * lr) + c[k]
    elif args.fed_strategy == 'feddyn':
        for k, v in model.named_parameters():
            c_i[k] -= args.alpha * (v - prev_param[k])

    w = copy.deepcopy(model.state_dict())
    dict_to_device(w, 'cpu')
    dict_to_device(c_i, 'cpu')
    return w, epoch_loss, c_i

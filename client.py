import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_grad(model):
    grad = {k: torch.zeros(v.shape).to(v.device) for k, v in model.state_dict().items()}
    return grad


def dict_to_device(dict, device):
    for k in dict.keys():
        dict[k] = dict[k].detach().to(device)


def local_train(args, setting, lr, c, model, dataloader, device):
    # hyperparameter setting
    K = 0
    if setting.c_i is None:
        c_i = zero_grad(model)
    else:
        c_i = setting.c_i
        dict_to_device(c_i, device)
    if c is None:
        c = zero_grad(model)
    dict_to_device(c, device)

    # optimizer and scheduler setting
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            # if args.fed_strategy == 'feddyn':
            #     for name, param in model.named_parameters():
            #         param.grad.data += args.alpha * (param.data - prev_param[name]) - c_i[name]
            optimizer.step()
            K += 1
            batch_loss += loss.item()
        epoch_loss += batch_loss
        scheduler.step()
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
    return w, epoch_loss, scheduler.get_last_lr()[0], c_i

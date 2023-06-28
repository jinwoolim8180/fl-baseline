import torch
import numpy as np
import time
import argparse
import copy
import random
import os
from torch.utils.data import DataLoader, Subset
from torch.multiprocessing import Manager, Process, Queue

from dataset import get_dataset
from models import get_model
from client import local_train
from test import test


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--spc', action='store_true', help='whether spc or not (default: dirichlet)')
    parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--n_channels', type=int, default=3, help="number of channels of imges")

    # optimizing arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--fed_strategy', type=str, default='fedavg', help="optimization scheme e.g. fedavg")
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for feddyn")
    parser.add_argument('--n_gpu', type=int, default=4, help="number of GPUs")
    parser.add_argument('--n_procs', type=int, default=1, help="number of processes per processor")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    args = parser.parse_args()
    return args


def train_clients(args, param_queue, return_queue, device, train_dataset, client_settings):
    # get model
    model = get_model(args=args, device=device)
    while True:
        # get message containing paramters
        param = param_queue.get()
        if param == "kill":
            # kill this process
            break
        else:
            # parameter setting
            lr = param['lr']
            model.load_state_dict(param['model_param'])
            sel_clients = param['sel_clients']
            c = param['c']
            for client in sel_clients:
                setting = client_settings[client]
                dataloader = DataLoader(Subset(train_dataset, setting.dict_user))
                local_train(args, setting, lr, c, model, dataloader, device)
            return_queue.put("done")
        del param
    del model


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # parse args and set seed
    args = args_parser()
    print("> Settings: ", args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # set device
    if torch.cuda.is_available():
        n_devices = min(torch.cuda.device_count(), args.n_gpu)
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False
    os.environ["OMP_NUM_THREADS"] = "1"
    num_processes = torch.multiprocessing.cpu_count()  # Number of available CPU cores

    # create dataset and model
    train_dataset, test_dataset, dict_users = get_dataset(args=args)
    global_model = get_model(args=args, device=devices[0])
    glob_param = copy.deepcopy(global_model.state_dict())

    # create client setting list.
    manager = Manager()
    client_settings = []
    for idx in range(args.n_clients):
        s = manager.Namespace()
        s.dict_user = dict_users[idx]
        s.c_i = None
        s.alpha = 1.
        client_settings.append(s)

    # start fl and create pool
    param_queues = []
    result_queues = []
    processes = []
    n_processes = n_devices * args.n_procs
    for i in range(n_devices):
        for j in range(args.n_procs):
            param_queue, result_queue = Queue(), Queue()
            p = Process(target=train_clients, args=(args, param_queue, result_queue, devices[i], train_dataset, client_settings))
            p.start()
            processes.append(p)
            param_queues.append(param_queue)
            result_queues.append(result_queue)

    client_all = list(range(args.n_clients))
    n_clients = int(args.frac * args.n_clients)
    c = None
    lr = args.lr
    for round in range(args.epochs):
        # randomly select clients
        random.shuffle(client_all)
        clients = client_all[:n_clients]

        # assign clients to processes
        assigned_clients = []
        n_assigned_client = n_clients // n_processes
        for i in range(n_processes):
            assigned_clients.append(clients[:n_assigned_client])
            del clients[:n_assigned_client]
        for i, rest in enumerate(clients):
            assigned_clients[i].append(rest)

        # start training
        for i in range(n_processes):
            param_queues[i].put({'model_param': copy.deepcopy(glob_param), 'lr': lr,
                                 'sel_clients': assigned_clients[i], 'c': c})

        # aggregate
        for i in range(n_processes):
            result = result_queues[i].get()

        # test
        global_model.load_state_dict(glob_param)
        test_acc, test_loss = test(args, global_model, test_dataset, devices[0])
        print(test_acc)

    # close the pool to release resources
    for i in range(n_processes):
        param_queues[i].put("kill")

    for p in processes:
        p.join()
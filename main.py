import torch
import numpy as np
import time
import argparse
import copy
import random
import os

from torch.multiprocessing import Process, Event, Queue, Manager


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--n_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--spc', action='store_true', help='whether spc or not')
    parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--n_channels', type=int, default=3, help="number of channels of imges")

    # optimizing arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer (default: SGD)")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--fed_strategy', type=str, default='fedavg', help="optimization scheme e.g. fedavg")
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for feddyn")
    parser.add_argument('--n_gpu', type=int, default=0, help="number of GPUs")
    parser.add_argument('--n_procs', type=int, default=0, help="number of processes per processor")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    args = parser.parse_args()
    return args


def train_clients(clients, train_queue, return_queue, event, device, train_dataset, args):
    # get model
    while True:
        # wait until event is set
        event.wait()
        msg = train_queue.get()
        if msg == 'kill':
            break
        else:
            # training
            print("yeah")

        # reset event
        event.clear()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # set device
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False
    os.environ["OMP_NUM_THREADS"] = "1"

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

    # create dataset

    # create client list
    clients = []

    # create training processes
    processes = []
    events = []
    train_queue = Queue()
    result_queue = Queue()
    for i in range(n_devices):
        for j in range(args.n_procs):
            e = Event()
            p = Process(target=train_clients, args=(clients, train_queue, result_queue, e, devices[i], train_dataset, args))
            p.start()
            processes.append(p)
            events.append(e)
            time.sleep(0.1)

    # start fl
    for round in range(args.epochs):
        for idx, p in enumerate(processes):
            e = events[idx]

    # shutdown processes
    for idx, _ in enumerate(processes):
        train_queue.put('kill')
        e.set()
    time.sleep(5)

    for p in processes:
        p.join()
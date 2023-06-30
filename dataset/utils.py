import numpy as np
from torchvision import datasets, transforms
from dataset.leaf import FEMNIST, ShakeSpeare


def get_dataset(args):
    # load dataset and split users
    if args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.n_clients = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.n_clients = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        elif args.dataset == 'fashion-mnist':
            trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                                transform=trans_fashion_mnist)
            dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                                transform=trans_fashion_mnist)
        elif args.dataset == 'cifar10':
            trans_cifar10_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trans_cifar10_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_test)
        elif args.dataset == 'cifar100':
            trans_cifar100_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trans_cifar100_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100_test)
        else:
            exit('Error: unrecognized dataset')
        # sample users
        if args.iid:
            dict_users = sampling_iid(dataset_train, args.n_clients)
        else:
            if args.spc:
                dict_users = sampling_spc(dataset_train, args.n_clients)
            else:
                dict_users = sampling_dirichlet(dataset_train, args.n_clients, beta=args.beta)

    return dataset_train, dataset_test, dict_users


def sampling_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    if dict_users == {}:
        return "Error"
    return dict_users


def sampling_spc(dataset, num_users):
    """
    2 shards per client sampling
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    if dict_users == {}:
        return "Error"
    return dict_users


def sampling_dirichlet(dataset, num_users, beta=0.2):
    """
    dirichlet sampling
    :param dataset:
    :param num_users:
    :param beta:
    :return: dict of image index
    """
    min_size = 0
    labels = np.array(dataset.targets)
    while min_size < 10:
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        for indices in [(labels == l).nonzero()[0] for l in np.unique(labels)]:
            indices = indices[np.random.permutation(len(indices))]
            p = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.zeros(num_users)
            for i in range(num_users):
                proportions[i] = p[i]*(len(dict_users[i]) < (len(labels) / num_users))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions)*len(indices)).astype(int)[:-1]
            splitted_indices = np.split(indices, proportions)
            for i in range(num_users):
                dict_users[i] = np.concatenate((dict_users[i], splitted_indices[i]))
        min_size = min([len(indices) for indices in dict_users.values()])
    if dict_users == {}:
        return "Error"
    return dict_users
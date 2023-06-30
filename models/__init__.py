from models.basenet import MLP, CNNMnist, CNNFemnist, CNNCifar, CharLSTM
from models.resnet import resnets

def get_model(args, device, img_size=[28, 28]):
    # set n_channels and n_classes
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        args.n_channels = 1
        args.n_classes = 10
    elif args.dataset == 'femnist':
        args.n_channels = 1
        args.n_classes = 10
    elif args.dataset == 'cifar10':
        args.n_channels = 3
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        args.n_channels = 3
        args.n_classes = 100
    
    # build model
    if args.model == 'cnn':
        if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
            net = CNNMnist(args=args).to(device)
        elif args.dataset == 'femnist':
            net = CNNFemnist().to(device)
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            net = CNNCifar(args=args).to(device)
    elif args.model == 'lstm':
        net = CharLSTM().to(device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.n_classes).to(device)
    else:
        if args.model not in resnets:
            exit('Error: unrecognized model')
        else:
            net = resnets[args.model](args=args).to(device)

    return net
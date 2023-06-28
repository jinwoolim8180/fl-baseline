from models.nets import MLP, CNNMnist, CNNFemnist, CNNCifar, CharLSTM

def get_model(args, device, img_size=[28, 28]):
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = CNNMnist(args=args).to(device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net = CNNFemnist(args=args).to(device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net = CharLSTM().to(device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.n_classes).to(device)
    else:
        exit('Error: unrecognized model')
    return net
import copy
import torch

def aggregate(args, w, w_glob, c, c_locals):
    # fedavg
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    # feddyn
    if args.fed_strategy == 'feddyn':
        for k in c.keys():
            c[k] -= args.alpha * (w_avg[k] - w_glob[k]) * len(w) / args.n_clients
        for k in w_avg.keys():
            w_avg[k] -= c[k] / args.alpha

    # scaffold
    elif args.fed_strategy == 'scaffold':
        c_avg = copy.deepcopy(c_locals[0])
        for k in c_avg.keys():
            for i in range(1, len(c_locals)):
                c_avg[k] += c_locals[i][k]
            c_avg[k] = torch.div(c_avg[k], len(c_locals))
        for k in c.keys():
            c[k] += len(c_locals) * c_avg[k] / args.num_users

    return w_avg, c
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test(args, model, dataset, device):
    model.eval()

    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.test_bs)
    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = model(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.item(), test_loss


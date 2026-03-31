import yaml
import argparse
import torch
import torchmetrics
from basis import get_base_matrix
from model  import SpectralModel
from utils import seed_everything, get_split, train_model



def main_worker(config, data_path = None, return_all = False):
    print(config)
    seed_everything(config['seed'])
    if config["cuda"]:
        device = "cuda"
    torch.device(device)

    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    hidden_dim = config['hidden_dim']
    feat_dropout = config['feat_dropout']
    dropout1 = config["dropout1"]
    power = config["power"]
    gamma =config["gamma"]
    base = config["base"]


    if data_path is None:
        data_path =  'data/train_data/{}.pt'.format(config["dataset"])
    e, u, x, y, adj = torch.load(data_path)
    e, u, x, y = e.cuda(), u.cuda(), x.cuda(), y.cuda()
    e = 1 - e

    V_orth = get_base_matrix(e, power + 1, base=base).to(dtype=torch.float32)

    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(y, nclass, config['seed']) 
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    
    ut = u.permute(1, 0)
    nfeat = x.size(1)

    net = SpectralModel(nclass, nfeat, hidden_dim, feat_dropout, dropout1= dropout1, power = power, V_orth=V_orth).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)

    net_args =[e, u, ut, x]

    res, best_val_acc, best_test_acc, best_state_dict = train_model(
        net, optimizer, evaluation, epoch, train, valid, test, y, net_args, early_stop=True, gamma=gamma)
    print(best_test_acc)
    print(res)

    return res, best_val_acc, best_test_acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', default='chameleon')
    parser.add_argument('--base', default="chebyshev")
    parser.add_argument('--save-path', default="None")
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden-dim', type=int, default=None, help='Number of hidden dimension.')
    parser.add_argument('--dropout1', type=float, default=None)
    parser.add_argument('--feat-dropout', type=float, default=None)
    parser.add_argument('--power', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--use-reg', type=int, default=0)
    
    args = parser.parse_args()

    if args.use_reg:
        configs_directory = "configs_reg"
    else:
        configs_directory = "configs"
    print(configs_directory)
    
    
    config = yaml.load(open(f'{configs_directory}/{args.base}_config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    config["base"] = args.base
    
    vars_args = vars(args)
    for hyperparameter in config.keys():
        if hyperparameter in vars_args and vars_args[hyperparameter] is not None:
            config[hyperparameter] = vars_args[hyperparameter]

    main_worker(config)

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-skew', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--personalized_learning_rate', type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')

    # pFedHN
    parser.add_argument('--n_hidden_layer', type=int, default=3, help="number of hypernet's hidden layers")
    parser.add_argument("--embed_dim", type=int, default=20, help="hypernet's embedding dim")
    parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet's hidden dim")
    
    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    
    args = parser.parse_args()
    cfg = dict()
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg
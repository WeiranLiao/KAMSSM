
import argparse
def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default='./datasets/', help="Path of datasets")

    parser.add_argument("--dataset", type=str, default="HW", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for residual link')

    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1, help='weight for residual link')

    parser.add_argument('--temp', type=float, default=0.01, help='weight decay')
    parser.add_argument('--eta', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lamda', type=float, default=1, help='weight decay')
    parser.add_argument('--delta', type=float, default=0.05, help='weight decay')


    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4 , help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times. Default is 10.")
    parser.add_argument("--save_results", action='store_true', default=True, help="xx")
    parser.add_argument("--save_all", action='store_true', default=True, help="xx")
    parser.add_argument("--save_loss", action='store_true', default= True, help="xx")
    parser.add_argument("--save_ACC", action='store_true', default=True, help="xx")
    parser.add_argument("--save_F1", action='store_true', default=True, help="xx")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")


    parser.add_argument("--num_epoch", type=int, default=200, help="Number of training epochs. Default is 200.")



    args = parser.parse_args()

    return args
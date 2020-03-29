import argparse

import torch

from network.net import LogisticRegression
from utils.data import loaders, data_size
from utils.model import train_model, load_model, predict_model

torch.manual_seed(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int,default=100,
                        help='Training  Batch size  default: 100')
    parser.add_argument('--iters', type=int, default=3000,
                        help='Number of training iterations  default: 3000')
    parser.add_argument('--lr', type=float,default=0.001,
                        help='Model learning rate  default: 0.001' )
    parser.add_argument('--load', type=bool,default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = LogisticRegression(784, 10)  # 1x784  X 784x10     =     1x10
    if args.load:
        model_name = 'model-lqhrb.pkl'
        load_model(model, 'weights/{}'.format(model_name))
        while True:
            predict_model(model)

    else:
        n_batches = int(data_size / args.batch)  # 600
        n_epochs = int(args.iters / n_batches)  # 5

        train_ds_loader, test_ds_loader = loaders(args.batch)

        train_model(model,
                    train_ds_loader, test_ds_loader,
                    args.lr,
                    n_epochs,
                    True)

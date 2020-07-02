import argparse


def train_parser(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir',
                        default='/content/img_align_celeba')
    parser.add_argument('--csv_file', dest='csv_file',
                        default='/content/processed_file.txt')
    parser.add_argument('--latent_size', dest='latent_size',
                        type=int, default=100)
    parser.add_argument('--epochs', dest='epochs',
                        type=int, default=100)
    parser.add_argument('--log_interval', dest='log_interval',
                        type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=1e-3)

    return parser.parse_args(args)

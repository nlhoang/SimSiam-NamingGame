import argparse
import random
import sys
import datetime
from pathlib import Path
from tempfile import mkdtemp
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import SimCLR, SimSiam, SimSiamVI, SimSiamVAE
from utils import param_count, CIFAR10Dataset, CIFAR100Dataset, ImageNet100Dataset, display_loss, Logger
from evaluation import extract_features, train_classifier, evaluate_classifier
IS_SERVER = False


def args_define():
    parser = argparse.ArgumentParser(description='Train SimSiamVAE models.')
    parser.add_argument('--eval-model', default='SimCLR', choices=['SimCLR', 'SimSiam', 'SimSiamVI', 'SimSiamVAE'])
    parser.add_argument('--dataset', default='CIFAR', choices=['CIFAR', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--backbone', default='cnn', choices=['cnn', 'resnet_cifar', 'resnet_torch'])
    parser.add_argument('--batch-size', type=int, default=128, help='batch size of model [default: 128]')
    parser.add_argument('--feature-dim', type=int, default=512, help='feature dim from backbone [default: 512]')
    parser.add_argument('--latent-dim', type=int, default=2048, help='latent dim from projector [default: 2048]')
    parser.add_argument('--variable-dim', type=int, default=1024, help='variable dim from predictor [default: 1024]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 500]')
    parser.add_argument('--eval-epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--interval-saved', type=int, default=50, help='interval for saving models [default: 100]')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--classifier', default='DeeperMLP', choices=['Linear', 'MLP', 'DeeperMLP'])
    parser.add_argument('--num-classes', type=int, default=10, choices=[10, 100], help='number of classes')
    parser.add_argument('--debug', type=bool, default=True, help='debug vs running')
    args = parser.parse_args()
    return args


def set_seeds(seed):
    if seed == -1:
        seed = random.randint(1, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed: {:.2g}'.format(seed))


def initialize(args):
    if args.dataset == 'CIFAR':
        args.backbone = 'cnn'
        args.feature_dim = 512
        args.latent_dim = 512
        args.variable_dim = 512
        args.num_classes = 10
    elif args.dataset == 'CIFAR100':
        args.backbone = 'resnet_cifar'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100
    else:  # args.dataset == 'ImageNet':
        args.backbone = 'resnet_torch'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100

    if args.debug:
        args.dataset = 'CIFAR'
        args.backbone = 'cnn'
        args.classifier = 'Linear'
        args.epochs = 2
        args.eval_epochs = 2
        args.feature_dim = 512
        args.latent_dim = 16
        args.variable_dim = 16
        args.num_classes = 10

    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(runPath))
    print('Expt:', runPath)
    print('RunID:', runId)
    return runPath


def evaluate(args, model, dataloader_train, dataloader_test, device):
    train_features, train_labels = extract_features(model, dataloader_train, device)
    print('Finish extract train features')
    test_features, test_labels = extract_features(model, dataloader_test, device)
    print('Finish extract test features')

    if device == torch.device('mps'):
        device = torch.device('cpu')
    classifier = train_classifier(classifier=args.classifier, features=train_features, labels=train_labels,
                                  num_classes=args.num_classes, device=device, epochs=args.eval_epochs)
    evaluate_classifier(classifier=classifier, features=test_features, labels=test_labels, device=device, max_k=5)


def main():
    set_seeds(-1)
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    path = '/data/' if IS_SERVER else '../data/'

    if args.eval_model == 'SimCLR':
        train = SimCLR.train
        model = SimCLR.SimCLR(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
    elif args.eval_model == 'SimSiam':
        train = SimSiam.train
        model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
    elif args.eval_model == 'SimSiamVI':
        train = SimSiamVI.train
        model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
    else:  # args.eval_model == 'SimSiamVAE':
        train = SimSiamVAE.train
        model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim, backbone=args.backbone)

    if args.dataset == 'CIFAR':
        train_dataset = CIFAR10Dataset(path=path, train=True)
        eval_dataset_train = CIFAR10Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = CIFAR10Dataset(path=path, train=False, augmented=False)
        saved = args.run_path + args.eval_model + '_CIFAR'
    elif args.dataset == 'CIFAR100':
        train_dataset = CIFAR100Dataset(path=path, train=True)
        eval_dataset_train = CIFAR100Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = CIFAR100Dataset(path=path, train=False, augmented=False)
        saved = args.run_path + args.eval_model + '_CIFAR100'
    else:  # args.dataset == 'ImageNet'
        train_dataset = ImageNet100Dataset(path=path, train=True)
        eval_dataset_train = ImageNet100Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = ImageNet100Dataset(path=path, train=False, augmented=False)
        saved = args.run_path + args.eval_model + '_ImageNet'

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_dataloader_train = DataLoader(eval_dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_dataloader_test = DataLoader(eval_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))

    # Training and Evaluating
    loss_history = train(model=model, dataloader=train_dataloader, learning_rate=args.learning_rate, device=device,
                         epochs=args.epochs, save_interval=args.interval_saved, save_prefix=saved)
    display_loss(loss_history, save_path=args.run_path+'loss.png')
    print('--------------------')
    evaluate(args=args, model=model, dataloader_train=eval_dataloader_train, dataloader_test=eval_dataloader_test, device=device)


if __name__ == "__main__":
    main()

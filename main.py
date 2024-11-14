import argparse
import random
import sys
import datetime
from pathlib import Path
from tempfile import mkdtemp
import torch
import numpy as np
from torch.utils.data import DataLoader
import SimSiamVI
import SimSiamVAE
import SimSiam
import SimCLR
from dataloader import FashionMNISTDataset, CIFAR10Dataset, SVHNDataset, CIFAR100Dataset, ImageNet100Dataset, TinyImageNetDataset
from utils import param_count, display_loss, Logger
from evaluation import extract_features, train_classifier, evaluate_classifier
IS_SERVER = False


def args_define():
    parser = argparse.ArgumentParser(description='Train SimSiamVAE models.')
    parser.add_argument('--eval-model', default='SimCLR', choices=['SimCLR', 'SimSiam', 'SimSiamVI', 'SimSiamVAE'])
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR', 'SVHN', 'CIFAR100', 'TinyImageNet', 'FashionMNIST', 'ImageNet'])
    parser.add_argument('--backbone', default='resnet_torch', choices=['cnn-img', 'cnn-mnist', 'resnet_cifar', 'resnet_torch'])
    parser.add_argument('--freeze_backbone', default=True, help='use freeze pretrained backbone')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size of model [default: 128]')
    parser.add_argument('--feature-dim', type=int, default=512, help='feature dim from backbone [default: 512]')
    parser.add_argument('--latent-dim', type=int, default=2048, help='latent dim from projector [default: 2048]')
    parser.add_argument('--variable-dim', type=int, default=256, help='variable dim from predictor [default: 1024]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 500]')
    parser.add_argument('--eval-epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--interval-saved', type=int, default=50, help='interval for saving models [default: 100]')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--classifier', default='DeeperMLP', choices=['Linear', 'MLP', 'DeeperMLP'])
    parser.add_argument('--num-classes', type=int, default=10, choices=[10, 100], help='number of classes')
    parser.add_argument('--debug', type=bool, default=False, help='debug vs running')
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
        args.backbone = 'resnet_torch'
        args.latent_dim = 512
        args.variable_dim = 256
        args.num_classes = 10
    elif args.dataset == 'SVHN':
        args.backbone = 'cnn-img'
        args.latent_dim = 512
        args.variable_dim = 256
        args.num_classes = 10
    elif args.dataset == 'FashionMNIST':
        args.backbone = 'cnn-mnist'
        args.latent_dim = 128
        args.variable_dim = 64
        args.num_classes = 10
    elif args.dataset == 'CIFAR100':
        args.backbone = 'resnet_torch'
        args.latent_dim = 512
        args.variable_dim = 256
        args.num_classes = 100
    elif args.dataset == 'ImageNet':
        args.backbone = 'resnet_torch'
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100
    else:  # args.dataset == 'TinyImageNet200':
        args.backbone = 'resnet_torch'
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 200

    if args.debug:
        # args.dataset = 'FashionMNIST'
        # args.backbone = 'cnn-mnist'
        # args.classifier = 'Linear'
        args.epochs = 2
        args.eval_epochs = 2
        args.latent_dim = 16
        args.variable_dim = 16
        args.num_classes = 10

    runId = args.eval_model + '-' + args.dataset + '-' + datetime.datetime.now().isoformat()
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
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(args)
    set_seeds(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    path = '/data/' if IS_SERVER else '../data/'

    if args.eval_model == 'SimSiamVAE':
        train = SimSiamVAE.train
        model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim,
                                      backbone=args.backbone, freeze_backbone=args.freeze_backbone)
    elif args.eval_model == 'SimSiam':
        train = SimSiam.train
        model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone, freeze_backbone=args.freeze_backbone)
    elif args.eval_model == 'SimSiamVI':
        train = SimSiamVI.train
        model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone, freeze_backbone=args.freeze_backbone)
    else:  # args.eval_model == 'SimCLR':
        train = SimCLR.train
        model = SimCLR.SimCLR(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone, freeze_backbone=args.freeze_backbone)

    if args.dataset == 'FashionMNIST':
        train_dataset = FashionMNISTDataset(path=path, train=True)
        eval_dataset_train = FashionMNISTDataset(path=path, train=True, augmented=False)
        eval_dataset_test = FashionMNISTDataset(path=path, train=False, augmented=False)
        saved = args.run_path + args.eval_model + '_CIFAR'
    elif args.dataset == 'CIFAR':
        train_dataset = CIFAR10Dataset(path=path, train=True, model='resnet_torch')
        eval_dataset_train = CIFAR10Dataset(path=path, train=True, augmented=False, model='resnet_torch')
        eval_dataset_test = CIFAR10Dataset(path=path, train=False, augmented=False, model='resnet_torch')
        saved = args.run_path + args.eval_model + '_CIFAR'
    elif args.dataset == 'SVHN':
        train_dataset = SVHNDataset(path=path, train=True)
        eval_dataset_train = SVHNDataset(path=path, train=True, augmented=False)
        eval_dataset_test = SVHNDataset(path=path, train=False, augmented=False)
        saved = args.run_path + args.eval_model + '_SVHN'
    elif args.dataset == 'CIFAR100':
        train_dataset = CIFAR100Dataset(path=path, train=True, model='resnet_torch')
        eval_dataset_train = CIFAR100Dataset(path=path, train=True, augmented=False, model='resnet_torch')
        eval_dataset_test = CIFAR100Dataset(path=path, train=False, augmented=False, model='resnet_torch')
        saved = args.run_path + args.eval_model + '_CIFAR100'
    elif args.dataset == 'TinyImageNet':
        train_dataset = TinyImageNetDataset(path=path, train=True, model='resnet_torch')
        eval_dataset_train = TinyImageNetDataset(path=path, train=True, augmented=False, model='resnet_torch')
        eval_dataset_test = TinyImageNetDataset(path=path, train=False, augmented=False, model='resnet_torch')
        saved = args.run_path + args.eval_model + '_TinyImageNet'
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

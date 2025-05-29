import argparse
import random
import sys
import datetime
from pathlib import Path
from tempfile import mkdtemp
import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
import SimSiamVI
import SimSiamVAE
import SimSiam
import SimCLR
import BYOL
from datasets import load_dataset
from base.dataloader import (FashionMNISTDataset, CIFAR10Dataset, CIFAR100Dataset,
                             ImageNet100Dataset, ImageNet100HFDataset, TinyImageNet200Dataset)
from base.utils import param_count, display_loss, Logger
from base.evaluation import extract_features, train_classifier, evaluate_classifier


def args_define():
    parser = argparse.ArgumentParser(description='Train SimSiamVAE models.')
    parser.add_argument('--eval-model', default='SSNG',
                        choices=['SimSiam', 'SimSiamVI', 'SSNG', 'BYOL', 'SimCLR'])
    parser.add_argument('--dataset', default='ImageNet100HF',
                        choices=['CIFAR', 'CIFAR100', 'ImageNet100', 'ImageNetTiny', 'FashionMNIST', 'ImageNet100HF'])
    parser.add_argument('--eval-epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--interval-saved', type=int, default=100, help='interval for saving models')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
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
    args.freeze_backbone = False
    args.num_workers = 32

    if args.dataset == 'FashionMNIST':
        args.image_size = 28
        args.backbone = 'cnn-mnist'
        args.feature_dim = 512
        args.latent_dim = 128
        args.variable_dim = 64
        args.num_classes = 10
        args.batch_size = 128
        args.learning_rate = 0.05
        args.epochs = 400
    elif args.dataset == 'CIFAR':
        args.image_size = 32
        args.backbone = 'resnet_cifar18'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 256
        args.num_classes = 10
        args.batch_size = 512
        args.learning_rate = 0.05
        args.epochs = 1000
    elif args.dataset == 'CIFAR100':
        args.image_size = 32
        args.backbone = 'resnet_cifar18'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 256
        args.num_classes = 100
        args.batch_size = 512
        args.learning_rate = 0.05
        args.epochs = 1000
    elif args.dataset == 'ImageNet100':
        args.image_size = 64
        args.backbone = 'resnet34'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 200
        args.batch_size = 256
        args.learning_rate = 0.05
        args.epochs = 500
    elif args.dataset == 'ImageNet100HF':
        args.image_size = 224
        args.backbone = 'resnet50'
        args.feature_dim = 2048
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100
        args.batch_size = 256
        args.learning_rate = 0.05
        args.epochs = 100
    elif args.dataset == 'ImageNetTiny':
        args.image_size = 224
        args.backbone = 'resnet18'
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 200
        args.batch_size = 256
        args.learning_rate = 0.05
        args.epochs = 100

    if args.debug:
        args.epochs = 2
        args.eval_epochs = 2

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
    classifier = train_classifier(features=train_features, labels=train_labels, num_classes=args.num_classes,
                                  device=device, epochs=args.eval_epochs)
    evaluate_classifier(classifier=classifier, features=test_features, labels=test_labels, device=device, max_k=5)


def get_model(args):
    if args.eval_model == 'SSNG':
        train = SimSiamVAE.train
        model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim,
                                      backbone=args.backbone, freeze_backbone=args.freeze_backbone)
    elif args.eval_model == 'SimSiam':
        train = SimSiam.train
        model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone,
                                freeze_backbone=args.freeze_backbone)
    elif args.eval_model == 'SimSiamVI':
        train = SimSiamVI.train
        model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone,
                                    freeze_backbone=args.freeze_backbone)
    elif args.eval_model == 'SimCLR':
        args.batch_size = 256
        train = SimCLR.train
        model = SimCLR.SimCLR(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone,
                              freeze_backbone=args.freeze_backbone)
    else:  # args.eval_model == 'BYOL':
        train = BYOL.train
        model = BYOL.BYOL(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone,
                              freeze_backbone=args.freeze_backbone)
    return model, train


def get_dataloader_eval(args):
    path = '../../MachineLearning/data/'
    if args.dataset == 'FashionMNIST':
        eval_dataset_train = FashionMNISTDataset(path=path, train=True, augmented=False)
        eval_dataset_test = FashionMNISTDataset(path=path, train=False, augmented=False)
    elif args.dataset == 'CIFAR':
        eval_dataset_train = CIFAR10Dataset(path=path, train=True, augmented=False, model='resnet_cifar')
        eval_dataset_test = CIFAR10Dataset(path=path, train=False, augmented=False, model='resnet_cifar')
    elif args.dataset == 'CIFAR100':
        eval_dataset_train = CIFAR100Dataset(path=path, train=True, augmented=False, model='resnet_cifar')
        eval_dataset_test = CIFAR100Dataset(path=path, train=False, augmented=False, model='resnet_cifar')
    elif args.dataset == 'ImageNet100':
        eval_dataset_train = ImageNet100Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = ImageNet100Dataset(path=path, train=False, augmented=False)
    elif args.dataset == 'ImageNet100HF':
        full_dataset = load_dataset("clane9/imagenet-100")
        eval_dataset_train = ImageNet100HFDataset(full_dataset['train'], train=False, augmented=False)
        eval_dataset_test = ImageNet100HFDataset(full_dataset['validation'], train=False, augmented=False)
    else: # args.dataset == 'ImageNetTiny':
        eval_dataset_train = TinyImageNet200Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = TinyImageNet200Dataset(path=path, train=False, augmented=False)

    dataloader_train = DataLoader(eval_dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=args.num_workers, persistent_workers=True)
    dataloader_test = DataLoader(eval_dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=args.num_workers, persistent_workers=True)
    return dataloader_train, dataloader_test


def get_dataloader_train(args):
    path = '../../MachineLearning/data/'
    if args.dataset == 'FashionMNIST':
        train_dataset = FashionMNISTDataset(path=path, train=True)
        saved = args.run_path + args.eval_model + '_FashMNIST'
    elif args.dataset == 'CIFAR':
        train_dataset = CIFAR10Dataset(path=path, train=True, model='resnet_cifar')
        saved = args.run_path + args.eval_model + '_CIFAR'
    elif args.dataset == 'CIFAR100':
        train_dataset = CIFAR100Dataset(path=path, train=True, model='resnet_cifar')
        saved = args.run_path + args.eval_model + '_CIFAR100'
    elif args.dataset == 'ImageNet100':
        train_dataset = ImageNet100Dataset(path=path, train=True)
        saved = args.run_path + args.eval_model + '_ImageNet100'
    elif args.dataset == 'ImageNet100HF':
        full_dataset = load_dataset("clane9/imagenet-100")
        train_dataset = ImageNet100HFDataset(full_dataset['train'], train=True, augmented=True)
        saved = args.run_path + args.eval_model + '_ImageNetHF100'
    else: # args.dataset == 'ImageNetTiny':
        train_dataset = TinyImageNet200Dataset(path=path, train=True)
        saved = args.run_path + args.eval_model + '_ImageNetTiny'

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers, persistent_workers=True)
    return dataloader, saved


if __name__ == "__main__":
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(args)
    set_seeds(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, train = get_model(args)
    train_dataloader, saved = get_dataloader_train(args)
    model.to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))

    pretrained = True
    if pretrained:
        trained_model_path = 'experiments/3-SSNG-ImageNet100HF-2025-05-28/SSNG_ImageNetHF100_100.pth'
        model.load_state_dict(torch.load(trained_model_path))
        print('Loading pretrained model from ' + trained_model_path)

    # Training and Evaluating
    loss_history = train(model=model, dataloader=train_dataloader, learning_rate=args.learning_rate, device=device,
                         epochs=args.epochs, save_interval=args.interval_saved, save_prefix=saved)
    display_loss(loss_history, save_path=args.run_path+'loss.png')
    print('--------------------')

    del train_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    print('Training dataloader cleared, GPU memory freed.')
    print('--------------------')

    eval_dataloader_train, eval_dataloader_test = get_dataloader_eval(args)
    evaluate(args=args, model=model, dataloader_train=eval_dataloader_train, dataloader_test=eval_dataloader_test,
             device=device)

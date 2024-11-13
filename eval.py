import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import SimSiamVI
import SimSiamVAE
import SimSiam
import SimCLR
from utils import FashionMNISTDataset, CIFAR10Dataset, SVHNDataset, CIFAR100Dataset, ImageNet100Dataset
from evaluation import extract_features, train_classifier, evaluate_classifier
IS_SERVER = True


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


def args_define():
    parser = argparse.ArgumentParser(description='Train SimSiamVAE models.')
    parser.add_argument('--eval-model', default='SimSiamVAE', choices=['SimCLR', 'SimSiam', 'SimSiamVI', 'SimSiamVAE'])
    parser.add_argument('--folder', type=str, required=True, help='The folder where the models are stored')
    parser.add_argument('--dataset', default='SVHN', choices=['CIFAR', 'SVHN', 'CIFAR100', 'ImageNet', 'FashionMNIST'])
    parser.add_argument('--backbone', default='cnn-img', choices=['cnn-img', 'cnn-mnist', 'resnet_cifar', 'resnet_torch'])
    parser.add_argument('--batch-size', type=int, default=128, help='batch size of model [default: 128]')
    parser.add_argument('--feature-dim', type=int, default=512, help='feature dim from backbone [default: 512]')
    parser.add_argument('--latent-dim', type=int, default=2048, help='latent dim from projector [default: 2048]')
    parser.add_argument('--variable-dim', type=int, default=1024, help='variable dim from predictor [default: 1024]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate [default: 1e-3]')
    parser.add_argument('--eval-epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--classifier', default='DeeperMLP', choices=['Linear', 'MLP', 'DeeperMLP'])
    parser.add_argument('--num-classes', type=int, default=10, choices=[10, 100], help='number of classes')
    parser.add_argument('--debug', type=bool, default=False, help='debug vs running')
    if IS_SERVER is False:
        sys.argv = ['script_name', '--eval-model', 'SimSiamVAE', '--folder', 'folder_name']
    args = parser.parse_args()
    return args


def initialize(args):
    if args.dataset == 'CIFAR':
        args.backbone = 'cnn-img'
        args.latent_dim = 256  # 512
        args.variable_dim = 128  # 256
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
        args.backbone = 'resnet-cifar'
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100
    else:  # args.dataset == 'ImageNet':
        args.backbone = 'resnet-torch'
        args.latent_dim = 2048
        args.variable_dim = 1024
        args.num_classes = 100

    if args.debug:
        args.classifier = 'Linear'
        args.eval_epochs = 2


def main(model_list):
    args = args_define()
    initialize(args)
    print(args)
    base_path = os.path.join('experiments', args.folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    path = '/data/' if IS_SERVER else '../data/'

    if args.dataset == 'FashionMNIST':
        eval_dataset_train = FashionMNISTDataset(path=path, train=True, augmented=False)
        eval_dataset_test = FashionMNISTDataset(path=path, train=False, augmented=False)
    elif args.dataset == 'CIFAR':
        eval_dataset_train = CIFAR10Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = CIFAR10Dataset(path=path, train=False, augmented=False)
    elif args.dataset == 'SVHN':
        eval_dataset_train = SVHNDataset(path=path, train=True, augmented=False)
        eval_dataset_test = SVHNDataset(path=path, train=False, augmented=False)
    elif args.dataset == 'CIFAR100':
        eval_dataset_train = CIFAR100Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = CIFAR100Dataset(path=path, train=False, augmented=False)
    else:  # args.dataset == 'ImageNet'
        eval_dataset_train = ImageNet100Dataset(path=path, train=True, augmented=False)
        eval_dataset_test = ImageNet100Dataset(path=path, train=False, augmented=False)

    eval_dataloader_train = DataLoader(eval_dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_dataloader_test = DataLoader(eval_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for model_name in model_list:
        print(model_name)
        model_path = os.path.join(base_path, model_name)

        if args.eval_model == 'SimSiamVAE':
            model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim,
                                          variable_dim=args.variable_dim, backbone=args.backbone)
        elif args.eval_model == 'SimSiam':
            model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
        elif args.eval_model == 'SimSiamVI':
            model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
        else:  # args.eval_model == 'SimCLR':
            model = SimCLR.SimCLR(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)

        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print('Model load successful')
        evaluate(args=args, model=model, dataloader_train=eval_dataloader_train, dataloader_test=eval_dataloader_test, device=device)
        print('--------------------')


list_SimSiamVAE_SVHN = ['SimSiamVAE_SVHN_50.pth', 'SimSiamVAE_SVHN_100.pth', 'SimSiamVAE_SVHN_150.pth', 'SimSiamVAE_SVHN_200.pth', 'SimSiamVAE_SVHN_final.pth']
list_SimSiam_SVHN = ['SimSiam_SVHN_50.pth', 'SimSiam_SVHN_100.pth', 'SimSiam_SVHN_150.pth', 'SimSiam_SVHN_200.pth', 'SimSiam_SVHN_final.pth']
list_SimSiamVAE_CIFAR100 = [
    'SimSiamVAE_CIFAR100_50.pth', 'SimSiamVAE_CIFAR100_100.pth', 'SimSiamVAE_CIFAR100_150.pth',
    'SimSiamVAE_CIFAR100_200.pth', 'SimSiamVAE_CIFAR100_250.pth', 'SimSiamVAE_CIFAR100_300.pth',
    'SimSiamVAE_CIFAR100_350.pth', 'SimSiamVAE_CIFAR100_400.pth', 'SimSiamVAE_CIFAR100_450.pth',
    'SimSiamVAE_CIFAR100_500.pth', 'SimSiamVAE_CIFAR100_final.pth']


if __name__ == "__main__":
    main(list_SimSiamVAE_SVHN)

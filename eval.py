from sklearn.neighbors import KNeighborsClassifier
from main import *


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


if __name__ == "__main__":
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(args)
    set_seeds(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, train = get_model(args)
    eval_dataloader_train, eval_dataloader_test = get_dataloader_eval(args)
    model.to(device)
    print('Model Size: {}'.format(param_count(model)))

    print('--------------------')
    trained_model_path = 'experiments/3-SSNG-ImageNet100HF-2025-05-28/SSNG_ImageNetHF100_200.pth'
    model.load_state_dict(torch.load(trained_model_path))

    train_features, train_labels = extract_features(model, eval_dataloader_train, device)
    print('Finish extract train features')
    test_features, test_labels = extract_features(model, eval_dataloader_test, device)
    print('Finish extract test features')
    classifier = train_classifier(features=train_features, labels=train_labels, num_classes=args.num_classes,
                                  device=device, epochs=args.eval_epochs)
    evaluate_classifier(classifier=classifier, features=test_features, labels=test_labels, device=device, max_k=5)

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(train_features, train_labels)
    knn_score = knn.score(test_features, test_labels) * 100
    print(f'KNN Score: {knn_score:.2f}%')

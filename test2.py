import torchvision
import torchvision.transforms as transforms
import torch

if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("dataD")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    print(0)




    print("train_loader:", train_loader)
    print("train_loader shape:", train_loader.shape)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(batch_idx, "\n")
        print(inputs.shape, targets.shape)
        print(type(targets))
        print(targets)
        exit()


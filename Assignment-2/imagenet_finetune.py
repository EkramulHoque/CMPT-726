import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import html_output as html
import numpy as np
from torch.utils.data import Sampler, SubsetRandomSampler

NUM_EPOCH = 2
valid_size = .1
random_seed = 0
divide_data = 100
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.50)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def train():
    ## Define the training dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', download=True, transform=transform)
    train_idx,test_idx=sampler(trainset, shuffle=True)
    print(' Number of training data: '+str(len(train_idx)),' Number of test data: '+str(len(test_idx)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              sampler=SubsetRandomSampler(train_idx), shuffle=False, num_workers=1, pin_memory= True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              sampler=SubsetRandomSampler(test_idx), shuffle=False, num_workers=1,
                                              pin_memory=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                         shuffle=False, num_workers=1,
    #                                           pin_memory=True)
    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    if torch.cuda.is_available():
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)
    ## Do the training
    precision_list = []
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #print('Train Image :', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if torch.cuda.is_available():
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        test_image, predicted_list, probability, correct, total ,precision= test(model, test_loader)
        precision_list.append(precision)
    print('Finished Training')
    web = html.generate_html(test_image, predicted_list, probability, len(train_idx), correct, total,precision_list)
    html.write_lines(web, "predict_table.html")

def test(model,test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    predicted_list = []
    test_image = []
    probability = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
           #print('Test Images: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_list.append(predicted)
            test_image.append(labels)
            probability.append(outputs.data)
            precision = 100 * correct / total
    print('Accuracy of the network : %d %%' % (precision))
    return test_image, predicted_list, probability, correct, total, precision


def sampler(dataset,shuffle):
    num_train = int(len(dataset)/ divide_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    return  train_idx,test_idx


if __name__ == '__main__':
    train()


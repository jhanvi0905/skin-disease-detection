import torch.utils.data as data
from sklearn.metrics import classification_report, accuracy_score
from src.process import prepare_train_data
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    """
    The class contaning neural network architecture and its flow of data
    """
    def __init__(self):
        """Initialize the model with following layers"""

        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.linear = torch.nn.Linear(32 * 31 * 31, 3)

    def forward(self, x):
        """Flow of data from layer to layer"""

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.linear(x.view(-1, 32 * 31 * 31))
        return x


def train(model, device, train_loader, optimizer, epoch, display=True):
    """Train the model for a given epoch"""

    model.train()
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if display:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(images), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", '--batch_size', type=int, default=4)
    parser.add_argument("-ep", '--epochs', type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-sp", "--split", type=float, default=0.8)
    parser.add_argument("-path", "--data_path", type=str , default='data/raw/')

    print("Loading Data-----------------------------")
    args = parser.parse_args()
    train_loader, valid_loader, train_labels, valid_labels = prepare_train_data(args.data_path, args.batch_size, args.split)
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nTraining Model---------------------------")
    for epoch in range(args.epochs):
        train(model, 'cpu', train_loader, optimizer, epoch)

    torch.save(model, 'models/256-'+str(args.batch_size)+'-'+str(args.epochs))
    print("\nModel Saved!")

    print("\nValidation Set Metrics--------------------")

    model.eval()
    y_pred_list = []
    y_targets = []
    with torch.no_grad():
        model.eval()
        for images, target in valid_loader:
            y_test_pred = model(images)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.extend(y_pred_tags.cpu().numpy())
            y_targets.extend(target.numpy())

    print("\nClassification Report")
    print(classification_report(y_targets, y_pred_list))
    print("\nAccuracy: " + str(accuracy_score(y_targets, y_pred_list)))
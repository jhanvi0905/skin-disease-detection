import torch.utils.data as data
from sklearn.metrics import classification_report, accuracy_score
from src.process import prepare_data_loader
from src.train_model import MyModel
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_report(model, test_image_path, batch_size):
    """Generate Report for given test set """

    test_loader, labels = prepare_data_loader(test_image_path, batch_size)
    y_pred_list = []
    y_targets = []
    with torch.no_grad():
        model.eval()
        for images, target in test_loader:
            y_test_pred = model(images)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.extend(y_pred_tags.cpu().numpy())
            y_targets.extend(target.numpy())

    print("\nClassification Report")
    print(classification_report(y_targets, y_pred_list))
    print("\nAccuracy: " + str(accuracy_score(y_targets, y_pred_list)))


if __name__ == "__main__":

    "Added defaults for batch, data and best model, incase of change specify it using the flag"
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", '--batch_size', type=int, default=4)
    parser.add_argument("-path", "--data_path", type=str, default='data/raw/')
    parser.add_argument("-mod", "--model", type = str, default ='models/256-5-10')

    args = parser.parse_args()
    print("Loading Model.............................")
    model = torch.load(args.model)

    print("Generating Report...........................")
    generate_report(model, args.data_path, args.batch_size)
import torch
import torch.nn as nn
import torch.optim as optim

from FaceMaskCNN import FaceMaskCNN


class FaceMaskTrainer:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = FaceMaskCNN().to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.epochs = 10

        print("Device:", self.device)

    def train_one_epoch(self, train_loader):
        self.model.train()  # sets dropout/batchnorm to training behavior

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()          # clear gradients from previous batch

            outputs = self.model(images)        # forward pass
            loss = self.criterion(outputs, labels)  # compute loss

            loss.backward()                     # backpropagation
            self.optimizer.step()               # update weights

            # accumulate stats
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc
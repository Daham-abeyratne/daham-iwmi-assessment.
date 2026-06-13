import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from FaceMaskCNN import FaceMaskCNN
from BasicPreprocessing import BasicPreprocessing


class FaceMaskTrainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FaceMaskCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Halves LR if val_loss doesn't improve for 2 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.epochs = 10
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": []
        }
        self.best_val_acc = 0.0
        self.class_names = None

        print(f"Device: {self.device}")

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def validate(self, val_loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total += labels.size(0)

        return running_loss / total, correct / total

    def fit(self, train_loader, val_loader, class_names=None):
        if class_names:
            self.class_names = class_names
        os.makedirs("models", exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc     = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{self.epochs}]  "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": self.class_names,
                }, "models/best_model.pth")
                print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print(f"\nTraining complete. Best val accuracy: {self.best_val_acc:.4f}")
        return self.history

    def plot_training_history(self, save_path="results/training_curves.png"):
        os.makedirs("results", exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History")

        ax1.plot(epochs, self.history["train_loss"], marker="o", label="Train")
        ax1.plot(epochs, self.history["val_loss"],   marker="o", label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

        ax2.plot(epochs, self.history["train_acc"], marker="o", label="Train")
        ax2.plot(epochs, self.history["val_acc"],   marker="o", label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved → {save_path}")

    def evaluate(self, test_loader, class_names=None, save_path="results/confusion_matrix.png"):
        os.makedirs("results", exist_ok=True)

        if os.path.exists("models/best_model.pth"):
            ckpt = torch.load("models/best_model.pth", map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if class_names is None:
                class_names = ckpt.get("class_names")
            print("Loaded best checkpoint for evaluation.")

        self.model.eval()
        all_preds, all_labels, correct, total = [], [], 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = torch.argmax(self.model(images), dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}  ({correct}/{total})")

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(
            cm,
            display_labels=class_names or ["Class 0", "Class 1"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix  (Test Acc: {test_acc:.4f})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved → {save_path}")

        return test_acc




import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder


class BasicPreprocessing:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_size = (128, 128)
        self.batch_size = 32
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.random_seed = 42

    def import_dataset(self):
        self.full_dataset = ImageFolder(self.dataset_path)
        self.class_names = self.full_dataset.classes
        self.class_to_idx = self.full_dataset.class_to_idx

        print("Dataset loaded successfully!")
        print("Classes:", self.class_names)
        print("Class mapping:", self.class_to_idx)
        print("Total images:", len(self.full_dataset))

    def inspect_dataset(self):
        total = len(self.full_dataset)
        print(f"Total images: {total}")

        stats = {}
        for class_name, label in self.class_to_idx.items():
            count = self.full_dataset.targets.count(label)
            print(f"{class_name}: {count}")
            stats[class_name] = count

        stats["total_images"] = total

        # Class balance check - useful to know if you need
        # weighted loss / class weights later
        print("\nClass balance:")
        for class_name in self.class_names:
            pct = (stats[class_name] / total) * 100
            print(f"{class_name}: {pct:.2f}%")

        return stats

    def create_transforms(self):
        # ImageNet mean/std - standard normalization values,
        # works well even though this isn't ImageNet data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Training transforms include augmentation so the model
        # generalizes better and doesn't just memorize the data
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Validation/test transforms have NO augmentation -
        # we want consistent, repeatable evaluation
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print("Transforms created successfully!")

    def split_dataset(self):
        total_size = len(self.full_dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size  # remainder avoids rounding loss

        generator = torch.Generator().manual_seed(self.random_seed)
        train_split, val_split, test_split = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        # Build two separate ImageFolder datasets (same images,
        # different transforms) so train gets augmentation and
        # val/test don't, while keeping the same split indices.
        train_dataset_full = ImageFolder(self.dataset_path, transform=self.train_transform)
        eval_dataset_full = ImageFolder(self.dataset_path, transform=self.val_transform)

        self.train_dataset = Subset(train_dataset_full, train_split.indices)
        self.val_dataset = Subset(eval_dataset_full, val_split.indices)
        self.test_dataset = Subset(eval_dataset_full, test_split.indices)

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Validation size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")

    def create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        print("DataLoaders created successfully!")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

    def visualize(self, num_images=6):
        images, labels = next(iter(self.train_loader))

        # Un-normalize so images display with correct colors/brightness
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        plt.figure(figsize=(12, 6))
        for i in range(min(num_images, len(images))):
            img = images[i] * std + mean
            img = img.permute(1, 2, 0).numpy().clip(0, 1)

            plt.subplot(2, 3, i + 1)
            plt.imshow(img)
            plt.title(self.class_names[labels[i]])
            plt.axis("off")

        plt.tight_layout()
        plt.show()

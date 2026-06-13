import torch
import torch.nn as nn

class FaceMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature Extraction Layers
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=128 * 16 * 16,
                out_features=256
            ),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                in_features=256,
                out_features=2
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = FaceMaskCNN()

    print(model)

    dummy_input = torch.randn(1, 3, 128, 128)

    output = model(dummy_input)

    print("\nOutput Shape:", output.shape)
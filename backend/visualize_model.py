import torch
import torch.nn as nn
from neutron import draw

# Example model: simplified OpenCLIP-like feature extractor
class CarFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CarFeatureNet()
dummy_input = torch.randn(1, 3, 224, 224)

# Generate visualization
draw(model, dummy_input, save_to="car_feature_net.png")
print("✅ Visualization saved as car_feature_net.png")

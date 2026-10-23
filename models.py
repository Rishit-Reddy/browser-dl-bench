"""Models under test. Random weights: only speed is measured, not accuracy."""
import torch, torch.nn as nn, torchvision

def cbr(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

class IFCRN(nn.Module):
    """Nucleus detector from Edgren Schüllerqvist (2025), sec. 6.1 / Fig. 13: a shallow U-Net with one skip connection.
    Not stated in the thesis (assumed here): 'same' padding and no output activation."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(*cbr(3, 32), *cbr(32, 32))
        self.pool = nn.MaxPool2d(2)
        self.mid = nn.Sequential(*cbr(32, 64), *cbr(64, 64))
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = nn.Sequential(*cbr(64, 32), *cbr(32, 32))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.enc(x)
        return self.out(self.dec(torch.cat([self.up(self.mid(self.pool(s))), s], 1)))

# name -> (constructor, input size). ResNet-18 is the baseline, IFCRN is the architecture of interest.
MODELS = {"resnet18": (lambda: torchvision.models.resnet18(weights=None), 224), "ifcrn": (IFCRN, 256),
          # classifier stand-ins: the group's papers use ResNet-family networks on 80x80 cell patches, healthy vs cancer (2 classes)
          "resnet18_80": (lambda: torchvision.models.resnet18(weights=None, num_classes=2), 80), "resnet50_80": (lambda: torchvision.models.resnet50(weights=None, num_classes=2), 80)}
BATCHES = (1, 8, 16)

if __name__ == "__main__":
    for n, (mk, s) in MODELS.items():
        m = mk().eval(); print(n, "params", sum(p.numel() for p in m.parameters()), "out", tuple(m(torch.randn(1, 3, s, s)).shape))

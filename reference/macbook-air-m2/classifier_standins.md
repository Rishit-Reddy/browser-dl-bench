| model | engine | batch 16 (ms/img) | batch 64 (ms/img) | batch 256 (ms/img) |
|---|---|---|---|---|
| resnet18_80 | PyTorch MPS (GPU) | 0.60 | 0.54 | 0.51 |
| resnet18_80 | Browser WebGPU (GPU) | 1.62 | 1.50 | 1.47 |
| resnet50_80 | PyTorch MPS (GPU) | 1.96 | 1.74 | 1.67 |
| resnet50_80 | Browser WebGPU (GPU) | 2.76 | 2.57 | 2.53 |

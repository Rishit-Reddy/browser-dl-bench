| model | engine | batch 1 (ms/img) | batch 8 (ms/img) | batch 16 (ms/img) |
|---|---|---|---|---|
| resnet18 | PyTorch CPU | 12.59 | 12.44 | 20.39 |
| resnet18 | PyTorch MPS (GPU) | 4.91 | 2.90 | 2.80 |
| resnet18 | Browser WASM (CPU) | 51.26 | 31.55 | 31.34 |
| resnet18 | Browser WebGPU (GPU) | 11.10 | 7.03 | 7.07 |
| ifcrn | PyTorch CPU | 60.70 | 113.23 | 55.67 |
| ifcrn | PyTorch MPS (GPU) | 8.25 | 7.38 | 7.17 |
| ifcrn | Browser WASM (CPU) | 58.55 | 62.18 | 65.06 |
| ifcrn | Browser WebGPU (GPU) | 14.70 | 12.61 | 12.55 |

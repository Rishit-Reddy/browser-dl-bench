| setting | ms/tile | GPU baseline MB | GPU peak MB | GPU growth MB | GPU after idle MB | renderer growth MB |
|---|---|---|---|---|---|---|
| batch 1, 256px | 14.5 | 514 | 688 | 174 | 672 | 510 |
| batch 2, 256px | 12.8 | 529 | 744 | 215 | 677 | 540 |
| batch 4, 256px | 12.4 | 572 | 920 | 348 | 869 | 499 |
| batch 8, 256px | 12.3 | 584 | 1133 | 549 | 1004 | 337 |
| batch 16, 256px | 12.3 | 506 | 1376 | 870 | 1059 | 349 |
| batch 32, 256px | 13.0 | 523 | 2917 | 2394 | 1059 | -14 |
| batch 64, 256px | 19.6 | 512 | 5162 | 4650 | 1032 | -343 |
| tile 128px, batch 1 | 6.0 | 549 | 765 | 216 | 680 | 506 |
| tile 256px, batch 1 | 13.8 | 583 | 774 | 191 | 723 | 516 |
| tile 512px, batch 1 | 56.2 | 506 | 874 | 368 | 820 | 252 |
| tile 1024px, batch 1 | 198.3 | 515 | 1360 | 845 | 1060 | 272 |
| tile 2048px, batch 1 | 1101.3 | 600 | 5237 | 4637 | 1068 | -410 |
| tile 4096px, batch 1 | FAILED: Error: failed to call OrtRun(). ERROR_CODE: 1, ERROR_MESSAGE | | | | | |

Long run, GPU process (footprint): 1394 MB after 25 batches, 1056 MB after 600; max 1394 MB.

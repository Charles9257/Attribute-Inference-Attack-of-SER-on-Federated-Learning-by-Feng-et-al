## SER performance of the federated model (test speakers, mean of 5 folds), Acc / UAR %

| FL | D_p | ours | paper (emobase) |
|---|---|---|---|
| FedSGD | EmoDB (paper: IEMOCAP) | 84.01 / 83.40 | 62.94 / 61.62 |
| FedSGD | CREMA-D (paper: CREMA-D) | 77.04 / 76.81 | 69.81 / 64.24 |
| FedSGD | RAVDESS (paper: MSP-Improv) | 73.54 / 70.97 | 54.17 / 47.23 |
| FedAvg | EmoDB (paper: IEMOCAP) | 86.90 / 86.90 | 61.22 / 61.25 |
| FedAvg | CREMA-D (paper: CREMA-D) | 79.22 / 79.04 | 71.26 / 66.38 |
| FedAvg | RAVDESS (paper: MSP-Improv) | 72.68 / 69.38 | 54.37 / 47.67 |

## Gender inference attack, first-layer updates (UAR %, mean of 5 private folds)

| FL | D_p | D_s (shadow) | ours | paper (emobase) |
|---|---|---|---|---|
| FedSGD | EmoDB | CREMA-D+RAVDESS | 46.64 | 89.33 (IEMOCAP) |
| FedSGD | CREMA-D | RAVDESS+EmoDB | 45.55 | 73.87 (CREMA-D) |
| FedSGD | RAVDESS | CREMA-D+EmoDB | 51.94 | 91.36 (MSP-Improv) |
| FedAvg | EmoDB | CREMA-D+RAVDESS | 43.22 | 87.45 (IEMOCAP) |
| FedAvg | CREMA-D | RAVDESS+EmoDB | 49.78 | 81.98 (CREMA-D) |
| FedAvg | RAVDESS | CREMA-D+EmoDB | 50.59 | 82.86 (MSP-Improv) |

## Which layer leaks most (attack UAR %)

| FL | D_p | first | second | last |
|---|---|---|---|---|
| FedSGD | EmoDB | 46.64 | 48.48 | 49.84 |
| FedSGD | CREMA-D | 45.55 | 52.56 | 50.29 |
| FedSGD | RAVDESS | 51.94 | 51.23 | 49.72 |
| FedAvg | EmoDB | 43.22 | 45.90 | 49.18 |
| FedAvg | CREMA-D | 49.78 | 51.36 | 48.25 |
| FedAvg | RAVDESS | 50.59 | 51.01 | 52.04 |

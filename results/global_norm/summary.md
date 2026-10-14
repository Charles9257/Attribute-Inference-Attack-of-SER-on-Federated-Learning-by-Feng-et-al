## SER performance of the federated model (test speakers, mean of 5 folds), Acc / UAR %

| FL | D_p | ours | paper (emobase) |
|---|---|---|---|
| FedSGD | EmoDB (paper: IEMOCAP) | 79.91 / 78.95 | 62.94 / 61.62 |
| FedSGD | CREMA-D (paper: CREMA-D) | 68.41 / 68.17 | 69.81 / 64.24 |
| FedSGD | RAVDESS (paper: MSP-Improv) | 65.00 / 62.06 | 54.17 / 47.23 |
| FedAvg | EmoDB (paper: IEMOCAP) | 82.90 / 81.30 | 61.22 / 61.25 |
| FedAvg | CREMA-D (paper: CREMA-D) | 69.10 / 68.78 | 71.26 / 66.38 |
| FedAvg | RAVDESS (paper: MSP-Improv) | 65.46 / 61.22 | 54.37 / 47.67 |

## Gender inference attack, first-layer updates (UAR %, mean of 5 private folds)

| FL | D_p | D_s (shadow) | ours | paper (emobase) |
|---|---|---|---|---|
| FedSGD | EmoDB | CREMA-D+RAVDESS | 47.37 | 89.33 (IEMOCAP) |
| FedSGD | CREMA-D | RAVDESS+EmoDB | 54.12 | 73.87 (CREMA-D) |
| FedSGD | RAVDESS | CREMA-D+EmoDB | 54.77 | 91.36 (MSP-Improv) |
| FedAvg | EmoDB | CREMA-D+RAVDESS | 55.84 | 87.45 (IEMOCAP) |
| FedAvg | CREMA-D | RAVDESS+EmoDB | 49.76 | 81.98 (CREMA-D) |
| FedAvg | RAVDESS | CREMA-D+EmoDB | 50.68 | 82.86 (MSP-Improv) |

## Which layer leaks most (attack UAR %)

| FL | D_p | first | second | last |
|---|---|---|---|---|
| FedSGD | EmoDB | 47.37 | 52.90 | 55.57 |
| FedSGD | CREMA-D | 54.12 | 55.49 | 57.95 |
| FedSGD | RAVDESS | 54.77 | 54.49 | 58.77 |
| FedAvg | EmoDB | 55.84 | 57.65 | 54.60 |
| FedAvg | CREMA-D | 49.76 | 55.54 | 54.99 |
| FedAvg | RAVDESS | 50.68 | 53.25 | 57.01 |

# Top-38 ROC coordinates

Each CSV contains `point`, `fpr`, `tpr`, and AUROC for one epsilon, curve (`real` or `synthetic`), and pathology. `real` is the real reference used by the corresponding epsilon evaluation. The combined file contains all 44 curves. The MATLAB file `top38_roc_coordinates_all.mat` has one curve per row with cell-array FPR/TPR vectors. Source files are the epsilon-specific `top38_xrv_roc_all.mat` and `label_agreement.json` files.

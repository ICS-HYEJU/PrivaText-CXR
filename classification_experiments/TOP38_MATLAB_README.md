# Top38 XRV AUROC MATLAB figure

Required files:

- `top38_xrv_roc_all_matlab.m`
- `RESULT_20261109_120931/evaluation_balanced/eps1/filtered_top38/top38_xrv_roc_all.mat`
- `RESULT_20261109_120931/evaluation_balanced/eps10/filtered_top38/top38_xrv_roc_all.mat`

From the repository root, run in MATLAB:

```matlab
run('classification_experiments/top38_xrv_roc_all_matlab.m')
```

The script creates a 3x4 figure for each epsilon arm. The first 11 tiles are
pathology ROC curves and the final tile contains the shared legend. It writes
`label_auroc_roc_all_paper_matlab.png` and `.fig` into each corresponding
`filtered_top38` directory.

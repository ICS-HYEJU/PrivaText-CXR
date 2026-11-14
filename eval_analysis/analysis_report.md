# Per-epsilon evaluation comparison

Result folders: eps1, eps10

## Comparison table

```
epsilon |    FID | FDS(sym) | FDS gen||real | FDS real||gen
--------+--------+----------+---------------+--------------
1.20233 | 7.9308 |  80.9541 |       57.7784 |      104.1298
 9.2673 | 8.2968 |  79.3523 |       59.0144 |       99.6902
```

> Direction: FID / FDS / LPIPS / CLIP gap are ↓ (lower better); SSIM / PSNR / CLIPScore are ↑ (higher better).
> Reliability order (§6.3): FID, FDS, LPIPS, CLIP gap > SSIM/PSNR (pixel proxies; generation ≠ reconstruction).

## eps1 vs eps10 direct comparison

```
       metric | direction |     eps1 |   eps10 |   delta | relative_change | better
--------------+-----------+----------+---------+---------+-----------------+-------
          FID |      down |   7.9308 |  8.2968 |  0.3661 |           4.62% |   eps1
     FDS(sym) |      down |  80.9541 | 79.3523 | -1.6018 |          -1.98% |  eps10
FDS gen||real |      down |  57.7784 | 59.0144 |  1.2360 |           2.14% |   eps1
FDS real||gen |      down | 104.1298 | 99.6902 | -4.4397 |          -4.26% |  eps10
```

## Model provenance

```
Model provenance (from ckpt_info.json):
  ε=1.20233  spent=1.2023 lora_rank=4 lora_alpha=4.0000 dir=eps1
  ε=9.2673   spent=9.2673 lora_rank=— lora_alpha=— dir=eps10
```

## Privacy–utility tradeoff

```
Privacy-utility tradeoff (expected as ε↓: quality worsens; §6.2):
  FID              rises with ε     ✗ UNEXPECTED (framework may be misbehaving or noise-limited)
  FDS(sym)         falls with ε     ✓ as expected
  FDS gen||real    rises with ε     ✗ UNEXPECTED (framework may be misbehaving or noise-limited)
  FDS real||gen    falls with ε     ✓ as expected
```

## FDS direction diagnosis

```
FDS direction diagnosis (mode-collapse vs hallucination; §6.5):
  ε=1.20233  mode-collapse (real||gen 1.80× gen||real)
  ε=9.2673   mode-collapse (real||gen 1.69× gen||real)
```

## Fair-comparison check

```
Fair-comparison check (§6.4):
  ✗ `n_gen` differs across models: eps1=408, eps10=361
```

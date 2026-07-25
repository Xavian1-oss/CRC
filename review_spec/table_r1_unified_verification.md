# Table R1 unified verification

Parsed **15** runs from `logs/review/r1u_*`.

## Metrics (G0 from baseline row in each log)

| Config | B1 MAE | B1 MSE | B1 NDR | B2 MAE | B2 MSE | B2 NDR | B3 MAE | B3 MSE | B3 NDR |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **G0** | 0.4108 | 0.3962 | — | 0.4654 | 0.4874 | — | 0.2184 | 0.1685 | — |
| **G1 always-on** | 0.4738 | 0.4665 | 8.2% | 0.5689 | 0.6328 | 10.6% | 0.2326 | 0.1803 | 33.9% |
| **G2 ridge+gate** | 0.4097 | 0.3941 | 97.9% | 0.4693 | 0.4920 | 97.1% | 0.2111 | 0.1626 | 90.0% |
| **G3 no dir** | 0.4079 | 0.3865 | 81.8% | 0.4684 | 0.4786 | 78.5% | 0.2098 | 0.1611 | 81.5% |
| **G4 deploy-only** | 0.4097 | 0.3941 | 62.1% | 0.4592 | 0.4838 | 77.8% | 0.2081 | 0.1584 | 79.4% |
| **G5 Full CRC** | 0.3914 | 0.3780 | 80.3% | 0.4392 | 0.4612 | 79.7% | 0.1903 | 0.1408 | 78.5% |

## Claim checks (rebuttal narratives)

- [PASS] **G1 vs G5 NDR+MAE (B1)**: G1 NDR=8.2% MAE=0.4738 vs G5 NDR=80.3% MAE=0.3914
- [PASS] **G1 vs G5 NDR+MAE (B2)**: G1 NDR=10.6% MAE=0.5689 vs G5 NDR=79.7% MAE=0.4392
- [PASS] **G1 vs G5 NDR+MAE (B3)**: G1 NDR=33.9% MAE=0.2326 vs G5 NDR=78.5% MAE=0.1903
- [PASS] **G2 conservative on B2 (MAE worse than G0)**: G2 MAE=0.4693 vs G0=0.4654, NDR=97.1%
- [PASS] **G5 vs G2 B1 MSE (Full CRC stronger global MSE)**: G5 MSE=0.3780 vs G2 MSE=0.3941 (MAE: G5=0.3914 G2=0.4097)
- [PASS] **G2 higher NDR than G5 on B1**: G2 NDR=97.9% vs G5=80.3%
- [PASS] **G5 NDR > G4 on B1 (representation helps deployment)**: G5 NDR=80.3% vs G4=62.1%
- [PASS] **G5 B2 MAE better than G0**: G5 MAE=0.4392 vs G0=0.4654
- [PASS] **G5 B1 MSE better than G0**: G5 MSE=0.3780 vs G0=0.3962
- [PASS] **G5 beats G4 on B3 MAE**: G5 MAE=0.1903 vs G4=0.2081

**Overall narrative support:** YES (direction gate: **setting-dependent**—B1 NDR G3 81.8% vs G5 80.3%; B2 NDR G5 79.7% vs G3 78.5%)

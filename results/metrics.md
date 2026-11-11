# 3R arm — simulation results

## Model validation

- SymPy vs closed form: max |ΔM| = 8.9e-16, |ΔC| = 1.8e-15, |ΔG| = 3.6e-15
- Passive swing with friction: energy loss 26.5872 J, friction dissipation 26.5872 J, max residual 9.27e-06 J

## Inverse kinematics

| Target [m] | Branch | Selected q [deg] | φ [deg] | w | Numerical IK q [deg] | w (numerical) | Numerical admissible? |
|---|---|---|---|---|---|---|---|
| (1.6, 0.6) | up | [ 69.9 -66.5 -43.7] | -40.2 | 1.6100 | [ 69.9 -66.5 -43.6] | 1.6100 | True |
| (-1.2, 0.9) | up | [ 174.3  -26.6 -109.3] | 38.3 | 1.2504 | [200.8 -78.7 -47.7] | 1.5607 | False |

## Trajectory planning (nominal inverse dynamics)

| Trajectory | T [s] | ∫τᵀτ dt | Mechanical work [J] | Peak τ1/τ2/τ3 [N·m] | Peak speed [rad/s] | Min height [m] | Min manipulability |
|---|---|---|---|---|---|---|---|
| Quintic | 3.000 | 1066.6 | 28.27 | 25.5/9.6/1.17 | 1.14 | 0.100 | 1.250 |
| Min-energy | 3.000 | 522.9 | 34.93 | 29.9/9.0/1.90 | 3.00 | 0.020 | 0.419 |
| Min-time | 0.847 | 1649.3 | 78.07 | 60.2/27.6/4.84 | 3.01 | 0.100 | 1.064 |

## Robustness — Nominal (plant = model)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PID | 1.71/1.23/1.68 | 3.66/2.86/3.17 | 57.6 | 133 | 15.93/4.76/1.31 | 31.8/10.0/1.9 | 36.7 | 1112 | 34 | 0.97 | 8 |
| CTC | 1.44e-08/3.73e-08/4.56e-08 | 7.36e-08/2.08e-07/2.54e-07 | 8.8e-07 | 4.38e-06 | 15.48/4.56/1.31 | 29.9/9.0/1.9 | 34.9 | 1048 | 30 | 0.15 | 20 |
| SMC | 1.48e-09/2.85e-09/5.97e-09 | 9.29e-09/2.86e-08/3.26e-08 | 6.91e-08 | 5e-07 | 15.48/4.56/1.31 | 29.9/9.0/1.9 | 34.9 | 1048 | 30 | 0.15 | 20 |

## Robustness — Payload +0.5 kg (0.5 kg point mass at the tool)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PID | 2.14/1.77/4.73 | 4.15/4.06/8.92 | 88.6 | 158 | 20.11/7.08/3.75 | 36.9/19.8/6.5 | 60.0 | 1875 | 61 | 0.73 | -31 |
| CTC | 0.223/1.22/6.52 | 0.366/2.92/13.6 | 62.1 | 111 | 19.33/6.94/3.69 | 34.7/19.6/6.2 | 57.5 | 1742 | 58 | 0.2 | -53 |
| SMC | 0.119/0.195/0.84 | 0.251/0.559/1.32 | 10.4 | 18.5 | 19.45/6.76/3.63 | 36.9/16.6/6.5 | 57.1 | 1748 | 57 | 0.22 | 9 |

## Robustness — Payload +1.0 kg (1.0 kg point mass at the tool)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PID | 2.59/2.4/8.03 | 4.68/6.28/15.2 | 127 | 232 | 24.42/9.73/6.34 | 40.9/30.2/10.6 | 85.3 | 2926 | 86 | 0.56 | -76 |
| CTC | 0.453/2.44/13.5 | 0.839/6.36/26.4 | 128 | 215 | 23.13/9.71/6.10 | 36.3/31.4/9.9 | 80.8 | 2666 | 83 | 0.2 | -128 |
| SMC | 0.237/0.396/1.8 | 0.513/1.21/4.14 | 22.1 | 49.9 | 23.58/9.41/6.02 | 44.3/28.9/9.8 | 81.4 | 2723 | 88 | 0.22 | -2 |

## Robustness — Disturbance (sinusoids + torque pulses)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PID | 1.65/1.38/1.65 | 3.95/2.75/3.21 | 57.4 | 139 | 15.88/5.39/1.35 | 34.9/11.2/2.4 | 36.8 | 1132 | 52 | 0.36 | 7 |
| CTC | 0.544/1.11/1.78 | 1.57/2.69/7.41 | 13.9 | 56.9 | 15.53/5.21/1.37 | 32.7/11.4/2.4 | 38.0 | 1081 | 50 | 0.045 | 25 |
| SMC | 0.0682/0.113/0.164 | 0.2/0.379/0.757 | 2.29 | 8.45 | 15.48/5.17/1.36 | 32.8/10.9/2.4 | 35.0 | 1073 | 49 | 0.055 | 20 |

## Robustness — Combined (1 kg payload, masses +10 %, friction +50 %, disturbance)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PID | 2.71/2.68/8.07 | 5.45/6.78/15.5 | 132 | 245 | 25.82/10.84/6.40 | 47.8/32.5/11.2 | 91.2 | 3301 | 106 | 0.35 | -80 |
| CTC | 0.603/2.76/13.1 | 1.32/7.08/25.7 | 127 | 215 | 24.57/10.76/6.07 | 42.8/33.5/10.2 | 90.0 | 3026 | 101 | 0.1 | -132 |
| SMC | 0.275/0.479/1.89 | 0.522/1.46/4.6 | 23.9 | 56.4 | 24.99/10.39/6.11 | 49.8/30.0/10.0 | 86.8 | 3079 | 105 | 0.14 | -2 |

## Robustness summary (tool-tip RMSE)

| Controller | Nominal [mm] | Combined [mm] | Mean over scenarios [mm] | Worst case [mm] | Worst scenario |
|---|---|---|---|---|---|
| PID | 57.6 | 132 | 92.5 | 132 | Combined |
| CTC | 8.8e-07 | 127 | 66.2 | 128 | Payload +1.0 kg |
| SMC | 6.91e-08 | 23.9 | 11.8 | 23.9 | Combined |

## Chattering study (sampled-data 1 kHz, Combined scenario)

| controller | RMSE θ1/θ2/θ3 [deg] | Max \|e\| θ1/θ2/θ3 [deg] | Tool RMSE [mm] | Tool max [mm] | RMS τ [N·m] | Peak τ [N·m] | E_mech [J] | ∫τᵀτ dt | TV [N·m/s] | HF power [%] | Min height [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SMC-sign | 0.111/0.166/0.853 | 0.216/0.611/3.88 | 10.4 | 43.5 | 42.38/20.03/8.01 | 68.1/28.7/9.9 | 187.2 | 9047 | 86081 | 91 | 17 |
| SMC | 0.275/0.48/1.9 | 0.522/1.47/4.63 | 24 | 56.7 | 24.99/10.40/6.11 | 49.8/30.0/10.0 | 86.9 | 3080 | 106 | 0.14 | -2 |
| SMC-tanh | 0.286/0.582/2.45 | 0.558/2.1/5.53 | 29.9 | 66.6 | 24.98/10.42/6.11 | 49.1/30.2/10.0 | 87.3 | 3079 | 105 | 0.14 | -9 |

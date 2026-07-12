# ALIGNN Work-Function Ray300 HPO Supporting Evidence Report

Generated: `2026-07-08 22:53:10 UTC`

Computing resource summary: 12 NVIDIA RTX 4090 GPUs, one GPU per trial.

## Summary

This report documents a systematic 300-trial Ray Tune hyperparameter optimization (HPO) experiment for ALIGNN on the work-function regression task. The HPO stage used a fixed 80/10/10 split and selected hyperparameters strictly by validation loss. The test split was not evaluated during HPO and was reserved for the final refit/test stage after the validation-selected configuration had been fixed.

In total, the HPO completed **300** Ray Tune trials and retained **5862** per-epoch validation records. **47** trials reached the full 50-epoch budget, while **253** trials were stopped early by ASHA. The retained training budget corresponds to approximately **117.2 full-budget trials**, with a budget utilization of **39.1%**.

The validation-selected best trial was `f770353c #198`. It reached the lowest validation loss `0.0496762` at epoch **50**, with validation MAE `0.3195 eV`, RMSE `0.4319 eV`, and R2 `0.8722`.

The selected configuration was:

```json
{
  "alignn_layers": 2,
  "batch_size": 8,
  "gcn_layers": 1,
  "hidden_features": 64,
  "loss": "smoothl1",
  "lr": 0.0008350182411692174,
  "weight_decay": 1.3314703565427604e-06
}
```

## Experimental Protocol

| Field | Value |
| --- | --- |
| Dataset | MatHub-2D work-function dataset |
| Target | workfunction_eV = evac_eV - efermi_eV |
| Model | ALIGNN scalar regression |
| Search algorithm | HyperOptSearch |
| Scheduler | ASHAScheduler (max_t=50, grace_period=8, reduction_factor=2) |
| Optimized metric | val_loss_model_space, minimize |
| Hardware | 12 NVIDIA RTX 4090 GPUs total, 1 GPU/trial |
| Split | 80/10/10 count protocol, n=1519/189/191 |
| Test-set guard | test split was not evaluated during HPO; used only after validation-selected final refit |

## Data and Target Definition

The prediction target was defined as:

`workfunction_eV = evac_eV - efermi_eV`

All HPO trials and the final refit/test used the same fixed split record: train/validation/test = 1519/189/191. Two negative work-function entries were retained and recorded in the split manifest. They were not removed post hoc, so the validation protocol is not affected by after-the-fact data cleaning.

| Split | n | mean/eV | std/eV | min/eV | median/eV | max/eV |
| --- | --- | --- | --- | --- | --- | --- |
| test | 191 | 5.3150 | 1.2585 | 1.4147 | 5.2013 | 8.4087 |
| train | 1519 | 5.2948 | 1.3652 | -12.67 | 5.3187 | 9.9456 |
| val | 189 | 5.3758 | 1.2112 | 1.6398 | 5.4369 | 8.1760 |

Number of negative-target entries recorded in the split manifest: `2`.

Both negative work-function entries were assigned to the train split; validation and test contain no negative target values.

![Target split distribution](../figures/fig9_target_split_distribution.png)

## Evidence Figure 0: HPO Workload Summary

![HPO workload summary](../figures/fig0_hpo_workload_summary.png)

This figure summarizes the number of proposed trials, retained per-epoch trajectories, ASHA early-stopping outcomes, and full-budget-equivalent training volume derived from the Ray Tune outputs.

## Search Space

| Parameter | Search range | Role |
| --- | --- | --- |
| hidden_features | choice[64, 128, 192, 256] | ALIGNN hidden width |
| alignn_layers | choice[1, 2, 3, 4] | angle-aware message-passing depth |
| gcn_layers | choice[1, 2, 3, 4] | atom graph convolution depth |
| lr | loguniform[3e-4, 2e-3] | AdamW max learning rate |
| weight_decay | loguniform[1e-6, 1e-4] | AdamW regularization |
| batch_size | choice[4, 8] | training mini-batch size |
| loss | choice[mse, smoothl1] | model-space regression loss |

The discrete part of the search space contains `4 x 4 x 4 x 2 x 2 = 256` architecture/training combinations, with two additional continuous log-uniform dimensions for `lr` and `weight_decay`. The HPO sampled all discrete values and covered learning rates from approximately `3e-4` to `2e-3` and weight decay values from approximately `1e-6` to `1e-4`.

## Evidence Figure 1: Trial-Level Search Cloud

![Trial-level search cloud](../figures/fig1_trial_level_search_cloud.png)

Each point represents the best validation loss reached by one hyperparameter configuration. The red star marks the validation-selected best trial; the orange dashed line is the top-5 mean `0.0517632`, and the gray dashed line is the all-trial median `0.124411`. The selected trial is drawn from a broad candidate distribution rather than from a small manual trial set.

## Evidence Figure 2: Full-Trial Performance Distribution

![Trial performance distribution](../figures/fig2_trial_performance_distribution.png)

This figure shows the distribution of best validation losses across all 300 trials, together with grouped views by hidden width and loss function. Lower-loss trials concentrate around `hidden_features=64` and `smoothl1`, consistent with the selected best configuration.

## Top-Trial Summary

| Rank | Trial | Best epoch | Val loss | Val MAE | Val RMSE | Val R2 | Status | Key hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | f770353c #198 | 50 | 0.0496762 | 0.3195 | 0.4319 | 0.8722 | full-budget | h=64, ALIGNN=2, GCN=1, lr=8.35e-04, wd=1.33e-06, bs=8, loss=smoothl1 |
| 2 | f0c92e93 #174 | 44 | 0.0505784 | 0.3204 | 0.4349 | 0.8704 | full-budget | h=64, ALIGNN=2, GCN=1, lr=7.03e-04, wd=2.72e-06, bs=8, loss=smoothl1 |
| 3 | 1da1807c #164 | 47 | 0.0514371 | 0.3249 | 0.4423 | 0.8659 | full-budget | h=64, ALIGNN=3, GCN=3, lr=8.62e-04, wd=4.98e-06, bs=8, loss=smoothl1 |
| 4 | 746894a0 #2 | 46 | 0.0525443 | 0.3139 | 0.4534 | 0.8591 | full-budget | h=128, ALIGNN=3, GCN=3, lr=1.93e-03, wd=2.72e-06, bs=8, loss=smoothl1 |
| 5 | 60367fb3 #74 | 43 | 0.0545799 | 0.3241 | 0.4609 | 0.8544 | full-budget | h=128, ALIGNN=2, GCN=1, lr=8.57e-04, wd=1.80e-05, bs=8, loss=smoothl1 |
| 6 | b807e368 #35 | 42 | 0.0548305 | 0.3095 | 0.463 | 0.8531 | full-budget | h=128, ALIGNN=1, GCN=3, lr=1.95e-03, wd=1.11e-05, bs=8, loss=smoothl1 |
| 7 | 671f32a0 #44 | 48 | 0.0557471 | 0.3154 | 0.4708 | 0.8481 | full-budget | h=256, ALIGNN=3, GCN=1, lr=4.29e-04, wd=4.57e-05, bs=8, loss=smoothl1 |
| 8 | 5d933394 #9 | 47 | 0.055886 | 0.3343 | 0.4681 | 0.8498 | full-budget | h=64, ALIGNN=1, GCN=3, lr=7.03e-04, wd=6.76e-05, bs=8, loss=smoothl1 |
| 9 | a88bc9d2 #114 | 49 | 0.0558947 | 0.3194 | 0.4717 | 0.8475 | full-budget | h=128, ALIGNN=2, GCN=1, lr=9.89e-04, wd=1.90e-06, bs=8, loss=smoothl1 |
| 10 | 4b40b9f2 #70 | 47 | 0.0559677 | 0.3346 | 0.4687 | 0.8495 | full-budget | h=64, ALIGNN=1, GCN=3, lr=7.04e-04, wd=1.68e-05, bs=8, loss=smoothl1 |

The complete top-20 table and the full-trial summary table are provided as `tables/top20_trials_by_validation_loss.csv` and `tables/trial_level_summary.csv`.

## Evidence Figure 3: Per-Trial Epoch Learning Trajectories

![Per-trial learning trajectories](../figures/fig3_epoch_learning_trajectories.png)

Gray curves show all trials, orange curves show the top 10% trials, and the red curve shows the validation-selected best trial. Short trajectories correspond to ASHA early stopping after the grace period, not missing records. The complete per-epoch trajectories are retained in `ray_hpo_epoch_trajectories.csv`.

## Evidence Figure 4: Ray Tune Best-So-Far Validation Loss

![Best-so-far validation loss](../figures/fig4_best_so_far_validation_loss.png)

The best-so-far curve records how the incumbent validation loss improved during the search. The final major update came from trial `f770353c #198` at epoch 50. The corresponding update table is provided as `tables/best_so_far_updates.csv`.

## Evidence Figure 5: ASHA Early Stopping and Budget Allocation

![ASHA budget allocation](../figures/fig5_asha_budget_allocation.png)

ASHA allocated the full 50-epoch budget to 47 trials and stopped 253 weaker candidates early. Thus, the 300 proposed configurations correspond to approximately 117.2 full-budget trials rather than 300 fully trained models.

## Evidence Figure 6: Hyperparameter Response Atlas

![Hyperparameter response atlas](../figures/fig6_hyperparameter_response_atlas.png)

The response atlas uses normalized validation regret to summarize how sampled hyperparameter values relate to validation performance. It is not a causal attribution analysis; rather, it shows that the report retains evidence over good and poor regions of the search space, not only the best configuration.

## Evidence Figure 7: Best-Config Search Boundary Check

![Best config boundary heatmap](../figures/fig7_best_config_boundary_heatmap.png)

| Parameter | Best value | Search-space position | Flag |
| --- | --- | --- | --- |
| alignn_layers | 2 | 0.333 | interior |
| batch_size | 8 | 1.000 | high-edge |
| gcn_layers | 1 | 0 | low-edge |
| hidden_features | 64 | 0 | low-edge |
| loss | smoothl1 |  | categorical |
| lr | 8.350e-04 | 0.54 | interior |
| weight_decay | 1.331e-06 | 0.0622 | low-edge |

The boundary check shows that the selected model is relatively small: `hidden_features=64` and `gcn_layers=1` lie at the lower end of the sampled range, while `batch_size=8` lies at the upper end of the candidate set. The learning rate is inside the sampled interval, and weight decay is near the lower end.

## Evidence Figure 8: Performance-Cost Frontier

![Performance-cost frontier](../figures/fig8_performance_cost_frontier.png)

This figure relates trial runtime to validation performance. The red curve marks the observed Pareto frontier and illustrates how ASHA concentrated training budget on promising configurations.

## Validation-Selected Best-Epoch Metrics

| Metric | Value |
| --- | --- |
| Selected trial id | f770353c #198 |
| Selected epoch | 50 |
| Validation loss | 0.0496762 |
| Validation MAE | 0.3195 eV |
| Validation RMSE | 0.4319 eV |
| Validation R2 | 0.8722 |

Selection metrics were extracted from the minimum validation loss over all retained epoch reports. For the selected best trial, the minimum occurred at epoch 50.

## Final refit/test

After the validation-selected best configuration was fixed, a final refit was performed and train/validation/test predictions were exported from the best validation checkpoint. The test split was used only at this final evaluation stage.

The final refit/test used the same fixed 80/10/10 split record as the HPO run. Hyperparameter selection was based only on validation loss, and test metrics were read only after the final refit was complete.

| Split | n | Loss | MAE/eV | RMSE/eV | R2 |
| --- | --- | --- | --- | --- | --- |
| train | 1519 | 0.021612 | 0.1673 | 0.496 | 0.8679 |
| val | 189 | 0.049676 | 0.3195 | 0.4319 | 0.8722 |
| test | 191 | 0.088359 | 0.4003 | 0.5858 | 0.7822 |

![Final refit parity/residual diagnostics](../figures/fig10_final_refit_parity_residual.png)

This figure reports parity and residual distributions for train, validation, and test splits, showing the behavior of the final refit model rather than an intermediate HPO trial.

![Final refit history](../figures/fig11_final_refit_history.png)

This figure shows the final refit validation loss/MAE/RMSE history and the checkpoint-selection basis.

## Reproducibility Materials

- HPO manifests: `ray_hpo_manifests.json`
- Global best trial: `global_best_trial.json`
- Full trial table: `ray_hpo_trial_table.csv`
- Per-epoch trajectories: `ray_hpo_epoch_trajectories.csv`
- Top-trial table: `tables/top20_trials_by_validation_loss.csv`
- Full-trial summary: `tables/trial_level_summary.csv`
- Best-so-far updates: `tables/best_so_far_updates.csv`
- Search boundary check: `tables/best_config_boundary_check.csv`
- Final refit outputs: final refit output folder

## Concise Interpretation

The additional 300-trial Ray Tune HPO experiment substantially improves the ALIGNN baseline relative to the originally reported ALIGNN value, and the final hyperparameters, split protocol, per-epoch trajectories, ASHA budget allocation, and final refit/test results are now explicitly documented.

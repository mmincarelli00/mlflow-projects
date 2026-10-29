RLF Prediction Pipeline: Files, Models and Test Results
29 ott 2026 · @mauro
Purpose and context
This is an experiment that tests failure-prediction models on radio links of a Radio Access Network based on O-RAN using synthetic data.
The scope is to analyze O-RAN RAN-level anomalies and implement failure prediction models on radio link failure (RLF) built on an open-source MLOps stack (MLflow).
The pipeline has three parts:
• synthetic_data.py creates plausible per-UE KPI data with known failures.
• features.py turns that data into labelled model inputs.
• train.py trains and compares four model families under identical conditions and logs every run to MLflow.
All results in this document come from synthetic data. None has been validated on real network data.
The three files
Each file does one job in a chain: generate data, build features and labels, then train and compare models.
synthetic_data.py: a stand-in for real KPM data
Scope: generate_dataset(n_ues=60, n_seconds=3600, n_cells=12, seed=7, episode_rate=1/400, sudden_rlf_rate=1/6000) returns one row per UE per second (216,000 rows in the default run).
Columns: ue_id, cell_id, t, rsrp, rsrq, sinr, neighbor_rsrp, ta, phr, ul_sinr, cqi, dl_bler, harq_retx, a3_events, ho_cmd, prb_util, connected_ues, a3_offset, hysteresis, ttt_ms, hour_of_day, and the failure flag rlf.
How it creates failures:
• Slowly wandering KPI signals form the normal background (an AR(1) helper, _ar1).
• Degradation episodes ramp the serving signal down over 10 to 45 s by 18 to 32 dB.
• Each episode ends in an RLF (55%), a handover (25%) or a recovery (20%).
• About 11% of RLFs are sudden, with no warning, so no model can predict them.
Known simplifications: ttt_ms, hour_of_day (constant 8 in the default run), prb_util and connected_ues carry no failure signal; recovery is abrupt; there is no spatial structure; and the default run has about 5.4 RLFs per UE-hour, far more than a real network.
Collecting the same dataset from a real system
A real network can supply the same per-UE, per-second table, but mainly through E2 reporting or vendor traces, because the standard 15-minute performance counters are far too coarse.
Where the data can come from:
• O-RAN near-RT RIC (E2 interface). An xApp subscribes to E2SM-KPM measurements (per cell and, depending on what the E2 node supports, per UE) and can receive RRC events through E2SM-RC. Records stream into a store such as Kafka or InfluxDB. This is the path closest to the target product.
• Vendor RAN traces. Call-trace and cell-trace exports from the vendor's OSS carry per-UE measurement reports, handover and failure events. 3GPP Minimization of Drive Tests (MDT) adds UE-reported RSRP, RSRQ and power headroom.
• SMO and the O1 interface. O1 delivers cell-level counters and configuration data. Counters usually arrive in 15-minute files, so use them for prb_util, connected_ues and handover parameters, not for second-level KPIs.
• Test beds and open datasets. srsRAN or OpenAirInterface with an E2 agent, Colosseum and OpenRAN Gym, POWDER, Arena, COSMOS and ColO-RAN produce the same kind of records without touching a live network.
• An operator design partner. The route to real RLF volume and realistic variety.
Where each column comes from:
Column
Typical real source
ue_id, cell_id, t
Pseudonymized RAN UE identifier, serving cell identity (NR cell ID or PCI), and a timestamp resampled to 1 s
rsrp, rsrq, sinr, neighbor_rsrp, a3_events
UE RRC measurement reports (serving and neighbor results, A3 event triggers) and CSI reports
cqi
UE CSI report
ta, phr
gNB-DU MAC/PHY: timing advance and the UE's power-headroom reports
ul_sinr, dl_bler, harq_retx
gNB-DU PHY/MAC statistics: uplink SINR, HARQ ACK/NACK results and retransmission counts
ho_cmd
Handover commands (RRC reconfiguration with synchronization) seen at the CU or in traces
prb_util, connected_ues
Cell-level counters (PRB usage and RRC-connected users, as defined in 3GPP TS 28.552)
a3_offset, hysteresis, ttt_ms
Handover (A3) measurement configuration, from configuration data or the RRC configuration
hour_of_day
Derived from the timestamp
rlf (the label)
Radio link failure events, defined below
The network does not see an RLF directly, because the UE declares it (for example when timer T310 expires). Network-side evidence includes RRC re-establishment requests, UE context releases with the cause "radio connection with UE lost", and UE RLF reports. Choose one operational definition and keep it fixed, since it defines the label.
How to build the CSV:
1. Pick a pilot cluster of cells, agree access and privacy terms, and pseudonymize UE identifiers (never store IMSI or phone numbers).
2. Collect for several weeks across busy and quiet periods. Real RLFs are far rarer than the synthetic 5.4 per UE-hour, so the number of RLF events, not rows, limits what can be learned.
3. Resample each UE to a contiguous 1-second grid and join per-UE records with cell counters and configuration by cell and time. Fill short gaps carefully and mark long ones, because make_labels assumes no missing seconds per UE.
4. Keep a UE's rows linked across handovers (the identifier often changes with the cell), so rolling features follow the UE.
5. Derive rlf from the chosen failure event.
6. Export one CSV with the same column names and run python train.py --csv real_kpm.csv --horizon 30.
Cautions:
• Per-UE, per-second reporting is not available from every vendor or E2 node. If a column such as ul_sinr, ta, phr or harq_retx is missing, remove it from features.py (the raw columns and the rolling list) and from the GRU's channel list, and bump FEATURE_VERSION.
• Positives will be much scarcer, so expect lower PR-AUC, and consider a longer horizon or more data.
• UE-level traces are personal data in most jurisdictions: pseudonymize and agree retention with the operator.
• Models trained on one vendor or cell type may need retraining elsewhere; the cell hold-out score is the early warning.
This mapping comes from general 3GPP and O-RAN knowledge and has not been checked against a specific vendor's data model.
features.py: features and labels
Scope: turns the raw rows into 105 model inputs plus a label (feature version rlf_features_v1).
Label: make_labels marks a row 1 if an RLF occurs for the same UE in the next horizon seconds, (t, t + horizon]. Every row in that window is positive, not just one. The failure second itself is not.
Features: all look backward, so nothing from the future leaks in. Rolling work is done per UE, and missing values at the start of a UE's history are set to 0.
Group
Count
Content
Raw columns
17
rsrp, rsrq, sinr, neighbor_rsrp, ta, phr, ul_sinr, cqi, dl_bler, harq_retx, a3_events, prb_util, connected_ues, a3_offset, hysteresis, ttt_ms, hour_of_day
Derived instantaneous values
2
nb_gap = neighbor_rsrp - rsrp; a3_margin = nb_gap - a3_offset - hysteresis
Rolling-window statistics
80
10 signals x 2 windows (5 s, 30 s) x 4 statistics (mean, std, min, slope)
Event and timing features
6
a3_count_5s, a3_count_30s, ho_count_30s, gap_pos_frac_30s, secs_since_a3, ta_drift_10s
The 10 rolling signals are rsrp, rsrq, sinr, ul_sinr, cqi, dl_bler, harq_retx, ta, phr and nb_gap. Identifiers and labels (ue_id, cell_id, t, rlf, ho_cmd, label) are never inputs.
Recognizing an RLF in real data
In real data the rlf column, and therefore every label, comes from evidence on both sides of the radio link, because the UE declares the failure and the network sees it only indirectly.
What happens at an RLF: the UE's physical layer reports "out of sync" repeatedly and starts timer T310. If the link does not recover before T310 expires, the UE declares a radio link failure. Random-access failure and reaching the maximum RLC retransmissions are other triggers. The UE then selects a cell and sends an RRC re-establishment request; if that fails, it falls back to idle and the connection is lost.
Evidence
Where it appears
What it tells you
RRC re-establishment request with a failure cause (for example "otherFailure")
gNB or CU RRC logs and traces
Best sign of an RLF the UE recovered from
UE context release with the cause "radio connection with UE lost"
NGAP, F1AP or XnAP messages, traces, abnormal-release counters (cause names vary by interface and vendor)
An RLF the UE did not recover from
UE RLF report
Reported by the UE on request (UE Information procedure)
Measurements at failure, failed cell, failure type and time of failure
UE-side logs
Drive-test and modem logs
T310 start and expiry, the most precise timestamp
Counters
OSS performance data
Re-establishment attempts and abnormal releases per cell: useful to check totals, too coarse to label a second
Pitfalls when labeling:
• The network sees it late. A re-establishment request arrives after T310 plus the UE's cell search, so the event can lag the true failure by hundreds of milliseconds to a few seconds. Labeling the arrival time biases measured lead times short; subtract the typical delay or use the failure time from the RLF report when available.
• Handover failures are different. A failure during handover (timer T304) also leads to re-establishment but has a different cause. Decide whether it counts as an RLF and keep the definition fixed.
• One failure can produce several records, such as a re-establishment request followed by a release. Merge events from the same UE within a few seconds into one RLF.
• Not every release is a failure. Normal releases, inactivity, network-initiated releases for load and a user switching off the phone should not count.
• Choose the scope: recovered RLFs only, or also those that end in a lost connection. For prediction, both are worth catching.
This describes general 3GPP behavior and has not been checked against a specific vendor's data model.
train.py: the comparison harness
Scope: trains the model families on identical splits, scores them with identical metrics and logs each family as one MLflow run (experiment oran-rlf-prediction).
What each run logs:
• Parameters, tags (dataset, feature version, horizon) and the averaged metrics.
• features.json, plus feature importances for XGBoost and the learned statistics for the z-score and the GRU.
• The trained model, named per family and horizon (for example rlf_xgboost_h30s).
Command-line options: --horizon, --n-ues, --seconds, --seed (data), --n-splits, --csv (real data with the same columns), --tracking-uri, --experiment, --no-gru, --gru-window, --gru-epochs.
Constants in the code: SEED = 42 (model randomness) and TARGET_FPR = 0.01 (false-alarm rate used to set the alert threshold).
Evaluation method
Every family is trained and scored on the same time-ordered splits, so differences come from the model and not from the data it saw.
Splits:
• 4 rolling-origin time folds. Train on the first 50%, 62.5%, 75% and 87.5% of the time range, then test on the next 12.5% after a gap (embargo) equal to the horizon. The gap stops the last training labels from reaching into the test period. Reported metrics are the average over the 4 folds.
• 1 cell hold-out. Train on the early period (before 70% of the time) using about 75% of the cells; test on the other 25% of cells in the later period. It checks generalization to unseen cells.
• 1 final model trained on all the data and logged to MLflow. It has no test score.
• Inside training, the GRU holds out the last 15% of its training period to choose the best epoch, and scikit-learn's HistGradientBoosting uses its own internal early-stopping validation when the data is large.
Alert threshold: each model outputs a risk score per UE-second. In every test fold the threshold is set at the 99th percentile of the scores of normal rows (TARGET_FPR = 0.01), so 1% of normal seconds raise an alert. This gives every model the same false-alarm budget. Because it is computed from the test labels, a live system would need a threshold fixed in advance.
Metrics:
Metric
What it measures
Better
pr_auc
Ranking quality over all thresholds (average precision); chance level equals the positive rate
Higher
recall_at_1pct_fpr
Share of pre-failure rows flagged at the 1% false-alarm threshold
Higher
event_recall
Share of RLF events with at least one alert in the horizon before them
Higher
median_lead_s
Median seconds between the first alert and the failure, over caught events
Longer, up to the horizon
alert_episodes_per_ue_hour
Separate alert bursts per UE-hour (alerts within one horizon are merged)
Lower
infer_us_per_sample
Scoring time per row in microseconds
Lower
cell_holdout_pr_auc
pr_auc on unseen cells
Higher
If TARGET_FPR is changed, the column name recall_at_1pct_fpr stays the same but the value refers to the new rate.
Model families
The four families run from the simplest rule to a deep-learning model, so each step has to earn its extra complexity.
Family
Learning type
Input
How it scores a row
Robust z-score
Unsupervised, labels ignored
6 engineered features
Largest absolute robust z-score among the 6 features
Isolation Forest
Unsupervised, but trained on normal rows only
All 105 features
Negative of scikit-learn's score_samples: short average path through random trees means more anomalous
Gradient boosting (XGBoost, or HistGradientBoosting if XGBoost is missing)
Supervised classifier
All 105 features
Predicted probability of an RLF within the horizon
GRU (PyTorch)
Supervised sequence model
Last 30 or 60 s of 13 raw KPI channels
Sigmoid of the network's output after reading the window
Why these four: the z-score shows how far a simple statistical rule goes; Isolation Forest is an anomaly detector that needs no failure labels; boosting is the standard strong learner for tabular data; and the GRU tests whether a sequence model reading raw KPIs adds anything.
Robust z-score baseline
The 6 features are rsrp_slope_5s, sinr_slope_5s, nb_gap, dl_bler_mean_5s, harq_retx_mean_5s and ta_slope_30s. The median and MAD (median absolute deviation) of each, computed on all training rows, are the only learned values (12 numbers). It has no randomness.
Isolation Forest
• Hyperparameters: n_estimators=200, contamination="auto" (logged but unused, because scores are read directly), random_state=SEED; scikit-learn defaults for the rest (256 rows per tree, all features).
• Learned: the structure of the 200 random trees (which feature and split value at each node).
• Idea: each tree splits the data at random; unusual rows are isolated in fewer splits.
Gradient boosting
• XGBoost hyperparameters: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sqrt(neg/pos), random_state=SEED.
• HistGradientBoosting fallback: max_iter=300, max_depth=6, learning_rate=0.05, class_weight="balanced"; its internal early stopping may keep fewer than 300 trees.
• Learned: the trees (split features, thresholds, leaf values). Each new tree corrects the errors of the previous ones, and their summed outputs go through a sigmoid.
• XGBoost versus the fallback: the same algorithm family with near-identical accuracy in these tests; XGBoost scores about 3 times faster at the 5 s horizon and 7 to 8 times faster at 30 s and gives feature importances.
GRU
• Hyperparameters: window 30 s by default (--gru-window), 1 layer with 32 hidden units, 12 epochs, batch 512, AdamW with learning rate 0.002 and weight decay 1e-4, gradient clipping at 1.0, positive-class weight sqrt(neg/pos).
• Learned parameters: 4,545 weights (GRU input weights 1,248, recurrent weights 3,072, biases 192, output layer 33).
• Idea: the GRU (Gated Recurrent Unit) reads the window second by second and keeps a memory vector, using an update gate and a reset gate to decide what to keep. The final memory goes through a linear layer and a sigmoid.
• Input: 13 raw channels (10 KPIs, nb_gap, a3_margin and ho_cmd). It does not see the 80 rolling features, because it is meant to learn its own temporal summaries. In fact GRU doesn't read the features columns but it reads directly the raw KPIs second by second, for example the last 30 to 60 seconds of RSRP, SINR, BLER and HARQ retransmissions, and after that it builds its own summary of how the signal has been evolving and outputs a risk score for “RLF within the horizon”.
• Training: inputs are standardized with training-set statistics, and the best epoch is chosen on a validation slice taken from the end of the training period.
How the GRU reads its input: window, channels, batch and epoch
The GRU is trained on blocks of shape (batch, window, channels), for example (512 windows, 60 seconds, 13 signals), and each term controls something different.
Term
Value
Meaning
Window
30 s by default, 60 s in some runs (--gru-window, logged as window_s)
The stretch of recent history read for one prediction: for second t, the rows from t-N+1 to t of the same UE, where N is the window value
Channels
13
One input signal measured at every second (the raw KPIs listed above). The same thing PyTorch calls input_size. The word keeps these raw signals apart from the 105 engineered features; it has nothing to do with radio channels.

A channel is the name used for the feature in GRU, this is in order to differentiate GRU features respect to the features of the other ML families, in fact a channel has a the raw value at every time step,
Batch
512
Set of 512 independent windows, randomly drawn from the training rows, each containing consecutive seconds of one UE, so a batch of 512 windows mixes many UEs and different moments in time.

512 are the windows that are scored before the weights are updated once. In the first time fold there are about 92,000 training windows, so one epoch is about 180 updates (92000/512), 1 update x batch
Epoch
12 at most
An epoch is a pass over all the training windows, which means many batches. A pass over a single batch is called an iteration (or step), and it is one weight update.
So an epoch is one full pass over all training windows. After each epoch the model is scored on the validation slice and the weights of the best epoch (by PR-AUC) are kept, so the final model may come from an earlier epoch.

The relationship is: windows per epoch = batch size × iterations per epoch.
Hidden size
32
The size of the GRU's memory vector. It does not depend on the input shape
Windows slide one second at a time, so consecutive windows overlap almost completely. They look backward only and never mix UEs. In the first seconds of a UE's data, the first available second is repeated to fill the window.
Window and horizon are independent. The horizon is set by the use case (how much warning an action needs); the window is set by how long the warning signs last, which in the synthetic data is a ramp of 10 to 45 s. A window longer than the horizon is common and fine, a shorter one also works, and equal values are only a coincidence of the defaults. In the tests, horizon 30 s with a 30 s window gave a PR-AUC of 0.441 and with a 60 s window 0.432, a difference within seed noise. A longer window costs more computation, since the GRU processes every second of it.
The label depends only on the horizon, never on the window. For the sample at t = 100 with a 60 s window and a 30 s horizon, the input is seconds 41 to 100 and the label is whether an RLF occurs in seconds 101 to 130. A 30 s window changes only the input; a 5 s horizon changes only the label. The train/test gap (embargo) equals the horizon, because labels look ahead; the window only reads the past and needs no gap.
Three different "windows" appear in this project:
Name
Where
Meaning
GRU window (30 or 60 s)
--gru-window
How much past the GRU reads as input
Rolling-feature windows (5 s and 30 s)
features.py
Spans over which means, standard deviations, minimums and slopes are computed for the tabular models
Horizon (5 or 30 s)
--horizon
How far ahead the label looks
Tabular models versus the sequence model
"Tabular" describes the shape of the input, not a type of algorithm: a table with one row per sample and a fixed set of columns. The z-score, Isolation Forest and gradient boosting all take one such row per prediction. The GRU takes a sequence.

Tabular models (z-score, Isolation Forest, boosting)
GRU
Input per prediction
One flat row: current values plus hand-built summaries of the last 5 and 30 s (105 features; 6 for the z-score)
The raw sequence of the last 30 or 60 s of 13 channels
How the past is represented
86 of the 105 features summarize it (rolling statistics, event counts, secs_since_a3 reaching back up to 600 s), designed by hand
Learned by the network from the raw sequence
Memory
None between rows: each row is scored independently from its own columns
A memory state carried through the window
The tabular models therefore do not predict from a single raw sample: their rows already encode the past. The test compares summaries designed by hand with summaries learned from raw sequences, so a GRU gain, if real, cannot be credited to "looking at history" alone. Other differences also matter: the tabular models receive 105 inputs including four noise columns while the GRU receives 13 raw channels (including ho_cmd), and a 60 s window reaches further back than the 30 s rolling features.
Results of the comparison tests
A supervised GRU scores highest on PR-AUC at the 30 s horizon, but only by about 0.03, and several other differences shrink or reverse when the seed changes.
All runs use the same synthetic dataset (60 UEs x 3,600 s = 216,000 rows, 105 features, 325 RLF events, data seed 7) and the same folds. Model seed is 42 unless stated. Metric names are explained in the evaluation section.
5 s horizon (positive rate 0.752%)
Model
PR-AUC
Recall at 1% FPR
Event recall
Median lead (s)
Alert episodes per UE-hour
Inference (us/row)
Cell hold-out PR-AUC
Robust z-score
0.291
0.552
0.694
4.75
7.454
0.210
0.364
Isolation Forest
0.184
0.337
0.527
3.75
5.984
9.31
0.210
HistGradientBoosting
0.343
0.591
0.777
5.00
7.957
6.68
0.410
XGBoost
0.314
0.604
0.787
5.00
7.724
2.265
0.443
Boosting beats the z-score, which beats Isolation Forest. XGBoost and HistGradientBoosting are effectively tied. Against chance (a PR-AUC equal to the positive rate), XGBoost is about 40 times better.
30 s horizon (positive rate 4.488%), model seed 42
Model
PR-AUC
Recall at 1% FPR
Event recall
Median lead (s)
Alert episodes per UE-hour
Inference (us/row)
Cell hold-out PR-AUC
Robust z-score
0.346
0.185
0.792
6.25
7.950
0.233
0.413
Isolation Forest
0.238
0.104
0.602
4.50
6.150
9.646
0.253
HistGradientBoosting
0.415
0.273
0.723
15.75
6.819
19.555
0.505
XGBoost
0.406
0.268
0.723
17.63
7.000
2.577
0.549
GRU (30 s window)
0.441
0.286
0.801
14.75
7.548
17.227
0.529
The longer horizon gives much more useful warning: the boosting models alert about 16 to 18 s ahead against about 6 s for the z-score. Lift over chance falls to about 9 times for XGBoost. Event recall is not comparable across horizons, because at 30 s chance alerts alone catch many events (the z-score reaches 0.792).
Repeat runs at 30 s: GRU window 60 and model seed 40
Run
PR-AUC
Recall at 1% FPR
Event recall
Median lead (s)
Alert episodes per UE-hour
Inference (us/row)
Cell hold-out PR-AUC
GRU, 60 s window, seed 42
0.432
0.275
0.758
15.88
7.145
44.474
0.578
Isolation Forest, seed 40
0.244
0.112
0.608
5.13
6.145
13.233
0.258
HistGradientBoosting, seed 40
0.403
0.269
0.741
15.63
7.031
21.216
0.507
XGBoost, seed 40
0.404
0.252
0.727
15.25
7.105
2.732
0.518
GRU, seed 40
0.434
0.280
0.715
15.25
6.943
35.015
0.524
The z-score row is identical in every run (0.346, 0.185, 0.792, 6.25, 7.95, cell hold-out 0.413) because it has no randomness. The window used in the seed-40 GRU run was not confirmed from the log.
What the tests show
• Model ranking: on PR-AUC, GRU (0.432 to 0.441) is ahead of XGBoost and HistGradientBoosting (0.403 to 0.415), then the z-score (0.346), then Isolation Forest (0.238 to 0.244). The GRU lead of about 0.03 appears in all three GRU runs and is larger than how far each boosting model's PR-AUC moved with the seed (0.002 to 0.012).
• At a fixed alert budget the gain is small: GRU recall at 1% FPR is 0.275 to 0.286 against 0.252 to 0.273 for boosting.
• Seed noise: changing only the model seed moved XGBoost cell hold-out from 0.549 to 0.518, XGBoost lead from 17.6 s to 15.3 s, XGBoost recall from 0.268 to 0.252, and HistGradientBoosting PR-AUC from 0.415 to 0.403 (likely because its early stopping uses a random validation split).
• Claims that did not hold up: that XGBoost gives the longest warning, that the GRU generalizes better to unseen cells, and that the GRU has the best event recall (it ranged from 0.715 to 0.801 across runs).
• Cost: the GRU scores a row in 17 to 44 us against 2.3 to 3.2 us for XGBoost, and needs PyTorch.
Conclusions and caveats
On this synthetic data, supervised learning clearly beats the unsupervised methods, the GRU adds a small unproven gain over gradient boosting, and XGBoost remains the sensible default.
What the evidence supports:
• Supervised models (boosting, GRU) beat the z-score and Isolation Forest on PR-AUC and on recall at a fixed false-alarm rate.
• XGBoost and HistGradientBoosting are equivalent in accuracy; XGBoost is faster and provides feature importances.
• A 30 s horizon gives roughly 15 to 18 s of lead time, against 4 to 6 s at a 5 s horizon, at the cost of lower lift over chance.
• The GRU improves PR-AUC by about 0.03 but changes the alert-budget results by only 0.01 to 0.03 and scores about 7 to 14 times slower than XGBoost.
Caveats:
• Synthetic data. The generator is simple and has about 5.4 RLFs per UE-hour, far above a real network, so absolute numbers are optimistic. About 11% of RLFs are sudden and cap event recall near 89%.
• Limited variation. All runs share one dataset (data seed 7, 325 events). The seed-40 run changed only model randomness. Differences below about 0.01 in PR-AUC, 0.03 in recall or cell hold-out and 2 s in lead time should be treated as noise.
• Oracle threshold. The alert threshold comes from the test fold's own normal rows. A deployed system would use a fixed threshold and perform somewhat worse.
• Unequal inputs. The tabular models see 105 engineered features; the GRU sees 13 raw channels, and a 60 s window gives it longer history than any rolling feature except secs_since_a3.
• GRU logging. Recent MLflow defaults to a PyTorch format that needs an input example, so the code asks for pickle and falls back to saving the weights. The pickle route worked on your machine; the fallback path has not been tested.
Next steps
The most useful next step is a multi-seed comparison, because the current GRU advantage is within what a single seed can produce.
1. Repeat with 3 to 5 model seeds and several data seeds (--seed) and report mean and standard deviation, so a difference can be judged against the spread.
2. Add a --target-fpr option and make the metric name follow its value.
3. Log each model's full hyperparameters (XGBoost currently logs only three) and log the z-score baseline as an MLflow pyfunc model so it appears in the Models view.
4. Add metrics: alert-episode precision, lead-time percentiles, and event recall counting only alerts with at least about 10 s of lead.
5. Add challengers: logistic regression as a cheap baseline, a GRU with a 15 s window, and a hybrid GRU that also receives the 105 engineered features.
6. Tune XGBoost (for example scale_pos_weight) and compare its feature importances at the 5 s and 30 s horizons.
7. Test on real data. Colosseum, OpenRAN Gym and ColO-RAN datasets, or KPM exports from a CSP design partner, are the decisive test of whether deep learning is worth its cost.

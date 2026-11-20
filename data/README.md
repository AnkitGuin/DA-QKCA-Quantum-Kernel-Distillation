# Data policy
Only synthetic IID mechanistic data was generated and used in executable pilots.
No original authors' dataset, CICIDS-2017, or UNSW-NB15 bytes are bundled.
Do not redistribute third-party datasets without checking license and permissions.
Synthetic data generator: `daqkca.data.loaders.synthetic`, seed 101..105.
Train/outer-val/final-test split 60/20/20; train-only preprocessing; split IDs and
hashes accompany each run. Random row splits are inappropriate for IDS flow leakage.

# Synthetic XOR card (experimented)
- Generator: i.i.d. Gaussian feature rows, d=12, n=360 per seed.
- Labels: `1{x0*x1 + 0.5*x2 - 0.25*x3 + 0.25*noise > 0}`.
- Classes: 2; sample counts and partitions are saved in each `splits.json`.
- Known relevant inputs: columns 0,1,2,3. Other columns are uninformative by design.
- Access: generated locally; no network or security-flow realism.
- Splits: stratified train 60%, validation 20%, locked final test 20%.
- Limitations: little imbalance; no temporal/group correlations; not an IDS benchmark;
  possible easy classical learnability; small n; random-seed variability.

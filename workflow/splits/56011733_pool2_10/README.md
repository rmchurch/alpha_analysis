# Pool-2-augmented split

This is a separate split namespace for the Pool-2 augmentation experiment.

- `train_folders.txt`: original 462 training cases plus 10 selected Pool 2 cases.
- `val_folders.txt`: the original 116 validation cases, unchanged.
- `later_folders.txt`: the original later/unseen list with the 10 added training cases removed.
- `pool2_added_to_train.txt`: the exact 10 Pool 2 cases added to training.

The ten Pool 2 cases are deterministic, sorted, and spread across the available
`G1600_00909+` cases so that the added training examples cover the Pool 2 range.

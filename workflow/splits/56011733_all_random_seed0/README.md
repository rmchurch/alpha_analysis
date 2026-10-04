# All-simulation random train/validation split

This split uses all 1,100 available simulations and shuffles them with
`random.Random(0)`, then assigns 880 simulations to training and 220 to
validation. The split is performed at the simulation level.

Pool counts:

- Train: 730 Pool 1 and 150 Pool 2.
- Validation: 178 Pool 1 and 42 Pool 2.

Pool 1 is `G1600_<0909`; Pool 2 is `G1600_0909+`. Both pools are present in
both training and validation.

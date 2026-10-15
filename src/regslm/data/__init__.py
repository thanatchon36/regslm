"""Train / dev / test / held-out splits and the numerically perturbed test set (paper §3.5).

Splits are grouped by article/section (D-024), with held-out rulebooks (D-023), a
post-cutoff slice (D-025) and a gold sample kept apart. ``perturb_set`` builds the
memorisation-control test set by perturbing numbers in unit text and remapping labels.
"""

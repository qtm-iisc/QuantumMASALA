# QuantumMASALA
QuantumMASALA: Quantum MAterialS Ab initio eLectronic-structure pAckage

This is a suite of codes which implement plane-wave pseudopotential based electronic structure calculations. The package includes an implementation of density functional theory (within the Kohn Sham framework) for solids, time-dependent density functional theory and many body perturbation theory (the GW Approximation).


## Installation Instructions:
`pip install --editable .`

**NOTE**: Do not pull files from the repo if you are using code older than 2022/09/27.
Slight changes to API have been made which might break your current work.

If you wanted to rollback:

`git checkout a1fe4b09f78cd6fb0356df61a73e553e3ae0ec52`


# RBFE Raw Data from SandboxAQ

This directory contains the raw data from RBFE calculations run by SandboxAQ on the "JACS set" JNK1, CDK2, and TYK2 targets:

- `ligands`: Structures and SMILES for the ligands
- `presto_ffs`: Force fields fit with `presto` and used for the calculations (with `ambertools` AM1-BCC charges).
- `raw_data`: csvs containing the raw results (for 3 repeat calculations of all edges).

Note that the TYK2 ligand `ejm_44` was removed from analysis as its experimental affinity is beyond the assay limit.
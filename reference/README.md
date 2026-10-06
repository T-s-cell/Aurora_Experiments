# reference/exp004 — read-only vendored copies of EXP-004 sources

These files are byte-for-byte copies from `VisionTS_Experiments/repro_ln_timesx_v1`
(and its read-only `timesx_data.py` dependency) at commit `dcd2425`, vendored so
that `verify_aggregate.py` can import the ORIGINAL aggregation module on machines
without the VisionTS tree (eta). They are never modified here.

Provenance (source -> copy, sha256 identical at vendoring time):

| source (VisionTS_Experiments) | copy | sha256 |
|---|---|---|
| repro_ln_timesx_v1/aggregate.py | exp004/aggregate.py | 1cd627f80f38d1e94504e155e6516cd0319bdc4e7f941939c9fe348da9e4124c |
| repro_ln_timesx_v1/common.py | exp004/common.py | 371c6f462e37fda0b26e1cb19c921b95ed234f56052a890f2d29d1b62b37dad7 |
| repro_zeroshot_ext/timesx_data.py | exp004/timesx_data.py | a27ec66e9b7e5c11bf95dba8bafc5f7d6b3cfdd6f2683a9774e5da80577a2cfa |
| repro_zeroshot_ext/timesx_data.py | repro_zeroshot_ext/timesx_data.py | a27ec66e9b7e5c11bf95dba8bafc5f7d6b3cfdd6f2683a9774e5da80577a2cfa |

The duplicate under repro_zeroshot_ext/ mirrors the original tree layout:
vendored exp004/common.py resolves its read-only dependency at
`../repro_zeroshot_ext/timesx_data.py` relative to itself, and that relative
path is preserved here so the import works unmodified.
| repro_ln_timesx_v1/configs/v1.yaml | exp004/configs/v1.yaml | 1a3710c9f837a6d65ffe74a716fa261360150d39764a0a6f0335345e8f26d140 |
| repro_ln_timesx_v1/results/overall.csv | exp004/results/overall.csv | 91a99d5701499062477d467f89820e451f1194ca5c780dedb51a0083dd85d2b9 |

verify_aggregate.py imports exp004/aggregate.py READ-ONLY (PYTHONDONTWRITEBYTECODE=1,
pure helpers only, never its main()) and compares against the ported aggregate.py
at full precision on data/Z0__test.npz.

Import side effect: exp004/common.py runs os.makedirs on its own-tree dirs
(manifests/checkpoints/logs/predictions/results/audit/dataset under exp004/).
These are import artifacts of the vendored copy inside THIS project (not the
original tree) and are git-ignored.

When /home/wlt/MMTS/VisionTS_Experiments exists (local), verify_aggregate.py
prefers the original tree directly; on other machines it falls back to this
vendored copy (sha256-verified).

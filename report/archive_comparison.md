# Supplied ZIP versus verified official run

Source ZIP SHA-256: `3ccb9e4bb4cba4f5fe3052af7142b21067851d8b4aca56ce974cdb1ee1296c48`. The ZIP is not tracked in Git.

| Check | Supplied ZIP summary | Verified official run |
|---|---:|---:|
| Status | final_refit_resource_violation | complete |
| Feasible Random Search trials | 10 | 23 |
| Feasible TPE trials | 0 | 31 |
| Selected decision | random_search #2 | random_search #35 |
| OOF F₂ | 0.7673 | 0.7742 |
| Test F₂ | 0.7486 | 0.7493 |
| Final latency ratio (cap 1.25×) | 1.327× | 0.951× |
| Final model size | 2.063 MB | 1.117 MB |

The ZIP README claimed 21 feasible Random Search and 23 feasible TPE trials, which disagreed with its summary JSON. The ZIP's final model violated the latency cap and is not described as feasible.

The verified run has a different latency protocol and a fresh output directory. Its test set is the same original holdout, which an older pilot had already viewed; the new optimization did not use pilot test metrics for selection.

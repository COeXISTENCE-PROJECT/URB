# Notes

- The study tests the current `default-pre-integration` route set, not clustering in
  general.
- The clustered set has 306 ODs, an average of 3.78 valid actions per OD, and mean
  cross-cluster edge overlap of 0.319 according to its stored summary.
- The standard route set is regenerated for each environment seed. This deliberately
  tests whether the fixed clustered set is a better practical base than ordinary seeded
  JanuX generation.
- Three paired seeds provide a local decision screen, not a publication-level estimate.
  The predeclared rule adds two seeds if the direction is inconsistent or too small.
- The local launch uses URB revision `d54bd65057701901e63aa7972ea48e43c7b604b7`
  from branch `tab` plus the uncommitted study files and baseline metadata repairs
  described below.
- Three replication suites were launched locally at `2026-08-12T16:08:29Z` with at
  most three SUMO processes active at once.
- That launch completed one AON arm per seed, then stopped in `analysis/metrics.py`
  because `baselines_clusters.py` had not written the required `algorithm` field.
  The completed simulations were retained; no travel data was regenerated.
- On `2026-08-14`, both baseline launchers were patched to record `algorithm`; the
  clustered launcher now also records its repository-relative script path. Metrics
  accepts old configs without this unused field, and the study runner forces headless
  plotting, resumes completed simulations, and refuses to overwrite partial outputs.
- Metrics for the three completed arms were recovered successfully. The other nine
  arms were relaunched at `2026-08-14T13:20:43Z` in three detached workers, preserving
  the predeclared seed pairing and counterbalanced route order.
- All 12 arms completed on `2026-08-15`. The final conclusion and paired values are
  reported in the study README, and the experiment folders now live under
  `studies/road-clustering-improves-iql/results/`.

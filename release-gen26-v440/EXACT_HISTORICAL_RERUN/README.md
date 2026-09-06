# EXACT_HISTORICAL_RERUN

Only physically available, unmodified historical artifacts are placed here.

## Important v440 provenance limitation

The active runtime used to assemble STAGE1 did **not** contain the byte-for-byte historical `tdc_gen26_integrated_v440.py` nor the previously referenced `TDC_Gen2_6_GitHub_Update_v440_v441.zip`. Therefore no reconstructed file is mislabeled as `EXACT_HISTORICAL_RERUN` for v440.

The package does include exact historical artifacts that were available in the runtime:

- experiment directories under `available_cwr_test/`;
- exported Gen2.5/Gen2.6 result files under `exported_results/`;
- exact frozen Gen2.4 v240 release archive under `legacy_release/`;
- additional historical Python source files physically present in the runtime.

The current executable v440-compatible implementation is separately labeled `CLEAN_ROOM_RECONSTRUCTION` and `CURRENT_CANONICAL_IMPLEMENTATION`.

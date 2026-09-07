# Native-v9 main code

This is the current native-v9 source snapshot with packaging-only path changes.
See the repository root README and docs/CODE_MAP.md for the maintained guide.
The original README is retained at ../../docs/native-v9-original-README.md;
its old GPU-guard description is historical and must not be used as a safety guarantee.

From the repository root, use scripts/test_cpu.sh for CPU tests and
scripts/run_native_v9.sh --dry-run to inspect a launch without starting it.
Do not run protect_gpu_lease.sh or historical scheduling scripts on a shared host.

Model implementations and loss functions are unchanged; source hashes and
packaging changes are recorded in ../../docs/source_manifest.json.

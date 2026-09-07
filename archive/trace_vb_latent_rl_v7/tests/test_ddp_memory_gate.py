import unittest

from tools.trace_policy_ddp_memory_smoke import (
    add_device_wide_headroom,
    memory_gate_failures,
)
from tools.full_supervisor import select_empty_gpus


def _snapshot(*, total, free, reserved, external):
    return {
        "phase": "synthetic",
        "device_total_mib": float(total),
        "device_free_mib": float(free),
        "device_used_mib": float(total - free),
        "this_rank_allocated_mib": float(max(0.0, reserved - 256.0)),
        "this_rank_reserved_mib": float(reserved),
        "external_or_nonallocator_mib": float(external),
    }


class DeviceWideMemoryGateTest(unittest.TestCase):
    def test_supervisor_preserves_requested_gpu_order(self):
        rows = [
            {"index": index, "memory_used_mib": used}
            for index, used in ((0, 1100), (4, 2400), (6, 2400), (7, 2300))
        ]
        self.assertEqual(
            select_empty_gpus(rows, 4, 2500, requested=[0, 4, 6, 7]),
            [0, 4, 6, 7],
        )

    def test_supervisor_waits_if_a_requested_gpu_is_over_threshold(self):
        rows = [
            {"index": index, "memory_used_mib": used}
            for index, used in ((0, 1100), (4, 2400), (6, 2501), (7, 2300))
        ]
        self.assertEqual(
            select_empty_gpus(rows, 4, 2500, requested=[0, 4, 6, 7]),
            [],
        )

    def test_known_gpu1_oom_shape_fails_despite_old_headroom_passing(self):
        # The failed run's old calculation reported 7469.7 MiB of room:
        # 24259.7 total - 16790 rank-local peak.  Charging the ~3.7 GiB held
        # by the background service and other rank contexts leaves <4 GiB.
        report = {
            "peak_cuda_memory_reserved_mib": 16790.0,
            "missing_trainable_gradients": [],
        }
        snapshots = [
            _snapshot(
                total=24259.6875,
                free=3707.6875,
                reserved=16790.0,
                external=3762.0,
            )
        ]
        add_device_wide_headroom(report, snapshots)

        self.assertGreater(24259.6875 - 16790.0, 4096.0)
        self.assertLess(report["effective_peak_device_headroom_mib"], 4096.0)
        self.assertTrue(memory_gate_failures(report))

    def test_lower_external_occupancy_can_pass_same_workload_peak(self):
        report = {
            "peak_cuda_memory_reserved_mib": 16790.0,
            "missing_trainable_gradients": [],
        }
        snapshots = [
            _snapshot(
                total=24259.6875,
                free=5303.6875,
                reserved=16790.0,
                external=2166.0,
            )
        ]
        add_device_wide_headroom(report, snapshots)

        self.assertGreater(report["effective_peak_device_headroom_mib"], 4096.0)
        self.assertEqual(memory_gate_failures(report), [])

    def test_missing_device_wide_measurement_fails_closed(self):
        report = {"missing_trainable_gradients": []}
        failures = memory_gate_failures(report)
        self.assertIn("missing:effective_peak_device_headroom_mib", failures)
        self.assertIn("missing:minimum_observed_device_free_mib", failures)

    def test_gradient_failure_is_part_of_same_gate(self):
        report = {
            "effective_peak_device_headroom_mib": 8192.0,
            "minimum_observed_device_free_mib": 8192.0,
            "missing_trainable_gradients": ["policy.weight"],
        }
        self.assertIn("missing_trainable_gradients", memory_gate_failures(report))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.engine.device_policy import DENSITIES, DeviceMode, DevicePolicy


class DevicePolicyTests(unittest.TestCase):
    def test_spread_is_explicit_round_robin(self) -> None:
        policy = DevicePolicy()
        self.assertEqual(
            [policy.device_for_stream(index) for index in range(4)],
            ["NPU", "GPU", "CPU", "NPU"],
        )

    def test_npu_toggle_drops_npu_from_priority(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.AUTO)
        snapshot = policy.toggle_npu()
        self.assertEqual(snapshot.active_devices, ("GPU", "CPU"))
        self.assertEqual(policy.device_for_stream(0), "AUTO:GPU,CPU")
        self.assertIn("NPU off", snapshot.label)

    def test_gpu_toggle_drops_gpu_in_npu_only(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.NPU_ONLY)
        snapshot = policy.toggle_gpu()
        self.assertEqual(policy.device_for_stream(0), "NPU")
        self.assertEqual(snapshot.active_devices, ("NPU",))
        self.assertIn("NPU_ONLY · NPU", snapshot.label)
        self.assertIn("GPU off", snapshot.label)

    def test_both_toggles_off_is_cpu(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.SPREAD)
        policy.toggle_npu()
        snapshot = policy.toggle_gpu()
        self.assertEqual(snapshot.active_devices, ("CPU",))
        self.assertEqual(policy.device_for_stream(0), "CPU")
        self.assertEqual(policy.device_for_stream(7), "CPU")
        self.assertEqual(snapshot.label, "OFF · NPU off · GPU off")

    def test_density_contract(self) -> None:
        policy = DevicePolicy()
        for density in DENSITIES:
            policy.set_density(density)
            self.assertEqual(policy.snapshot().density, density)
        with self.assertRaises(ValueError):
            policy.set_density(3)

    def test_cpu_only_and_split(self) -> None:
        policy = DevicePolicy(mode=DeviceMode.CPU_ONLY)
        self.assertEqual(policy.device_for_stream(0), "CPU")
        self.assertEqual(policy.snapshot().active_devices, ("CPU",))
        self.assertEqual(policy.snapshot().label, "CPU_ONLY · CPU")
        policy.set_mode(DeviceMode.SPLIT)
        self.assertEqual(policy.device_for_stream(0, "NPU"), "NPU")
        self.assertEqual(policy.device_for_stream(0, "GPU"), "GPU")

    def test_unavailable_device_is_not_offered(self) -> None:
        policy = DevicePolicy(available_devices=("GPU", "CPU"))
        modes = []
        for _ in range(len(tuple(DeviceMode))):
            modes.append(policy.cycle_mode().mode)
            if policy.mode == DeviceMode.AUTO:
                break
        self.assertNotIn(DeviceMode.NPU_ONLY, modes)
        with self.assertRaises(ValueError):
            policy.set_mode(DeviceMode.NPU_ONLY)


if __name__ == "__main__":
    unittest.main()

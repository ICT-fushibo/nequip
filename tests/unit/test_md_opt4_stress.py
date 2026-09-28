import unittest
import torch
from nequip.md_stages.stress_vjp import ForceStressEnergyVJP
from md_benchmark.stress_test_support import assert_replay_stable
from tests.test_opt4_stress_capture import PeriodicQuadratic, strain_inputs


class NequIPStressTests(unittest.TestCase):
    def test_cuda_strain_vjp_replay(self):
        for dtype in (torch.float32, torch.float64):
            for padding in (False, True):
                data = strain_inputs(dtype, "cuda:0", padding)
                model = ForceStressEnergyVJP(
                    PeriodicQuadratic(), data["pos"], data["cell"]
                ).eval()

                def body():
                    energy, forces = model(data)
                    return energy, forces, model.last_stress

                assert_replay_stable(body)
                self.assertEqual(float(model.strain.abs().max()), 0)


if __name__ == "__main__":
    unittest.main()

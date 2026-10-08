"""Setup-only source-scatter certification; runnable without e3nn/checkpoint."""
import importlib.util
from pathlib import Path
import unittest

import torch


path = Path(__file__).resolve().parents[2] / "nequip/md_stages/opt4_fusion.py"
spec = importlib.util.spec_from_file_location("source_vjp_production", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TP(torch.nn.Module):
    def forward(self, x, attr, weight):
        return x * attr * weight


class NequIPSourceVJPTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        # Deliberate cancellation, with repeated live source/destination IDs.
        x = torch.ones(2, 1, dtype=dtype)
        attr = torch.tensor([[16.0], [-16.0], [0.00085]], dtype=dtype)
        weight = torch.ones_like(attr)
        src = torch.zeros(3, dtype=torch.long)
        dst = torch.tensor([1, 1, 0])
        detail = {}
        boundary = module._Uniform1DTPScatter(TP(), dst, 2, detail)
        args = (x, attr, weight, dst, src)
        probe = [torch.ones_like(x)]
        expected = torch.tensor([[0.00085], [0.0]], dtype=dtype)
        return boundary, args, probe, expected, detail

    def test_certifies_repeated_source_sum_against_independent_terms(self):
        boundary, args, probe, expected, detail = self.fixture()
        actual = expected.clone()
        actual[0] += 1.274e-6
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        self.assertTrue(boundary.validate_source_vjp(actual, expected, args, 0, probe))
        self.assertEqual(detail["source_vjp_certificates"][0]["max_source_multiplicity"], 3)

    def test_rejects_both_wrong_values_against_independent_terms(self):
        boundary, args, probe, expected, _ = self.fixture()
        # Once strict parity fails, neither side may be trusted as ground truth.
        actual = expected + 0.01
        with self.assertRaises(AssertionError):
            boundary.validate_source_vjp(actual, actual + 4e-6, args, 0, probe)

    def test_live_destination_cotangent_is_required(self):
        boundary, args, probe, expected, _ = self.fixture()
        probe[0][1] = 2
        # Change only one of the cancelling edge terms, exposing wrong dst use.
        args[1][1] = 0
        expected[0] = 16.00085
        with self.assertRaises(AssertionError):
            boundary.validate_source_vjp(expected + 0.1, expected, args, 0, probe)

    def test_other_inputs_and_float64_retain_strict_contract(self):
        boundary, args, probe, expected, _ = self.fixture()
        for index in (1, 2):
            self.assertFalse(boundary.validate_source_vjp(expected + 1, expected, args, index, probe))
        self.assertFalse(boundary.validate_source_vjp(expected.double(), expected.double(), args, 0, probe))

    def test_padding_zeros_and_nonfinite_do_not_get_a_free_pass(self):
        boundary, args, probe, expected, _ = self.fixture()
        args[2].zero_()
        expected.zero_()
        for bad in (1e-4, float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                boundary.validate_source_vjp(expected + bad, expected, args, 0, probe)


if __name__ == "__main__":
    unittest.main()

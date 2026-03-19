from __future__ import annotations

import unittest

from verifier import BoxVerifier


class PhysicsVerifierTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            1: {"l": 100, "w": 100, "h": 100, "fragile": False},
            2: {"l": 100, "w": 100, "h": 100, "fragile": False},
            3: {"l": 100, "w": 100, "h": 100, "fragile": True},
        }
        self.verifier = BoxVerifier((300, 300, 300))

    def test_valid_floor_layout(self):
        placement = [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 100, "y": 0, "z": 0},
            {"id": 3, "x": 200, "y": 0, "z": 0},
        ]
        ok, msg = self.verifier.verify(self.items, placement)
        self.assertTrue(ok, msg)

    def test_overlap_rejected(self):
        placement = [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 50, "y": 0, "z": 0},
            {"id": 3, "x": 200, "y": 0, "z": 0},
        ]
        ok, msg = self.verifier.verify(self.items, placement)
        self.assertFalse(ok)
        self.assertIn("overlap", msg.lower())

    def test_floating_rejected(self):
        placement = [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 0, "y": 0, "z": 150},
            {"id": 3, "x": 200, "y": 0, "z": 0},
        ]
        ok, msg = self.verifier.verify(self.items, placement)
        self.assertFalse(ok)
        self.assertIn("support", msg.lower())

    def test_partial_support_rejected(self):
        placement = [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 50, "y": 0, "z": 100},
            {"id": 3, "x": 200, "y": 0, "z": 0},
        ]
        ok, msg = self.verifier.verify(self.items, placement)
        self.assertFalse(ok)
        self.assertIn("not fully supported", msg.lower())

    def test_fragile_support_rejected(self):
        placement = [
            {"id": 3, "x": 0, "y": 0, "z": 0},
            {"id": 1, "x": 0, "y": 0, "z": 100},
            {"id": 2, "x": 200, "y": 0, "z": 0},
        ]
        ok, msg = self.verifier.verify(self.items, placement)
        self.assertFalse(ok)
        self.assertIn("fragile", msg.lower())


if __name__ == "__main__":
    unittest.main()

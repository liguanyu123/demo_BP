from __future__ import annotations

import unittest

from trusted_solver import (
    _ranked_candidates,
    default_order_items,
    default_score_candidate,
    solve_with_policy,
)


class TransportStabilityPolicyTests(unittest.TestCase):
    def test_fragile_prefers_floor_candidates_when_available(self):
        items = {
            1: {"l": 200, "w": 200, "h": 120, "fragile": False},
            2: {"l": 100, "w": 100, "h": 80, "fragile": True},
            3: {"l": 100, "w": 100, "h": 80, "fragile": False},
        }
        placement = [{"id": 1, "x": 0, "y": 0, "z": 0}]
        ranked = _ranked_candidates(items, placement, 2, (300, 200, 250), default_score_candidate)
        self.assertTrue(ranked)
        self.assertTrue(all(cand["z"] == 0 for cand in ranked))

    def test_early_base_build_prefers_floor_for_non_fragile_items(self):
        items = {
            1: {"l": 200, "w": 200, "h": 120, "fragile": False},
            2: {"l": 100, "w": 100, "h": 90, "fragile": False},
            3: {"l": 100, "w": 100, "h": 80, "fragile": False},
            4: {"l": 100, "w": 100, "h": 70, "fragile": True},
            5: {"l": 90, "w": 90, "h": 60, "fragile": False},
            6: {"l": 80, "w": 80, "h": 60, "fragile": False},
        }
        placement = [{"id": 1, "x": 0, "y": 0, "z": 0}]
        ranked = _ranked_candidates(items, placement, 2, (300, 200, 260), default_score_candidate)
        self.assertTrue(ranked)
        self.assertTrue(all(cand["z"] == 0 for cand in ranked))

    def test_default_order_builds_base_without_permanently_sending_fragile_last(self):
        items = {
            1: {"l": 200, "w": 200, "h": 140, "fragile": False},
            2: {"l": 250, "w": 200, "h": 100, "fragile": True},
            3: {"l": 250, "w": 200, "h": 100, "fragile": False},
        }
        order = default_order_items(items)
        self.assertLess(order.index(2), order.index(1))
        self.assertLess(order.index(3), order.index(2))

    def test_solve_with_policy_keeps_fragile_on_floor_when_floor_slot_exists(self):
        items = {
            1: {"l": 200, "w": 200, "h": 120, "fragile": False},
            2: {"l": 100, "w": 100, "h": 80, "fragile": True},
            3: {"l": 100, "w": 100, "h": 80, "fragile": False},
        }
        solution = solve_with_policy(items, (300, 200, 250), default_order_items, default_score_candidate, heavy=False)
        self.assertEqual(len(solution), 3)
        by_id = {p["id"]: p for p in solution}
        self.assertEqual(by_id[2]["z"], 0)


if __name__ == "__main__":
    unittest.main()

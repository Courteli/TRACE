import unittest

import torch

from src.models.trace_exchangeable import (
    ExchangeablePathSeedMap,
    build_local_ranking_triplets,
    local_ranking_surrogate,
    merge_accuracy_guarded_gradients,
    trace_path_distance,
)


class ExchangeableSeedTests(unittest.TestCase):
    def test_seed_map_is_parameter_free_and_permutation_equivariant(self):
        mapper = ExchangeablePathSeedMap(
            seed_dim=4,
            n_steps=3,
            hidden_size=7,
            basis_seed=11,
        )
        self.assertEqual(list(mapper.parameters()), [])
        self.assertEqual(mapper.state_dict(), {})

        seeds = torch.randn(5, 4)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        self.assertTrue(
            torch.allclose(
                mapper.latent_noise(seeds[permutation]),
                mapper.latent_noise(seeds)[permutation],
            )
        )
        self.assertTrue(
            torch.allclose(
                mapper.controls(seeds[permutation]),
                mapper.controls(seeds)[permutation],
            )
        )
        self.assertTrue(
            torch.equal(
                mapper.latent_noise(torch.zeros_like(seeds)),
                torch.zeros(5, 3, 7),
            )
        )


class PathMetricTests(unittest.TestCase):
    def test_path_distance_is_symmetric_and_identity_preserving(self):
        left = torch.randn(4, 8, 16)
        right = torch.randn(4, 8, 16)
        identity = trace_path_distance(left, left)
        self.assertTrue(torch.allclose(identity, torch.zeros_like(identity), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                trace_path_distance(left, right),
                trace_path_distance(right, left),
                atol=1e-6,
            )
        )
        zeros = torch.zeros_like(left)
        self.assertTrue(
            torch.allclose(
                trace_path_distance(zeros, zeros),
                torch.zeros(left.shape[0]),
                atol=1e-6,
            )
        )

    def test_path_distance_uses_ordered_transitions(self):
        path = torch.zeros(1, 4, 3)
        path[0, 0, 0] = 1.0
        path[0, 1, 1] = 1.0
        path[0, 2, 2] = 1.0
        path[0, 3, 0] = -0.5
        reversed_path = path.flip(dims=[1])
        self.assertGreater(float(trace_path_distance(path, reversed_path).item()), 0.1)


class LocalRankingTests(unittest.TestCase):
    @staticmethod
    def _constant_paths(vectors):
        return torch.tensor(vectors, dtype=torch.float32).view(-1, 1, 2).repeat(1, 4, 1)

    def test_only_identifiable_mixed_groups_are_ranked(self):
        paths = self._constant_paths(
            [
                [1.00, 0.00],
                [0.98, 0.04],
                [0.00, 1.00],
                [0.04, 0.98],
                [0.95, 0.12],
                [-1.00, 0.00],
            ]
        )
        labels = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32)
        triplets = build_local_ranking_triplets(
            paths,
            labels,
            group_size=6,
            margin=0.08,
        )
        self.assertEqual(len(triplets), 2)
        # The hard wrong path near route A must use a route-A positive neighbor,
        # not collapse route A and route B into a global centroid.
        hard = min(triplets, key=lambda item: item.wrong_distance)
        self.assertIn(hard.correct_index, {0, 1})
        self.assertIn(hard.peer_index, {0, 1})

        self.assertEqual(
            build_local_ranking_triplets(
                paths[:4],
                torch.ones(4),
                group_size=4,
                margin=0.08,
            ),
            [],
        )
        self.assertEqual(
            build_local_ranking_triplets(
                paths[:4],
                torch.tensor([1, 0, 0, 0]),
                group_size=4,
                margin=0.08,
            ),
            [],
        )

    def test_frozen_mining_surrogate_matches_active_hinge_gradient(self):
        torch.manual_seed(7)
        paths = torch.randn(4, 5, 6, requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        triplets = build_local_ranking_triplets(
            paths.detach(),
            labels,
            group_size=4,
            margin=10.0,
        )
        self.assertTrue(all(item.active for item in triplets))

        exact = paths.sum() * 0.0
        for item in triplets:
            exact = exact + 10.0
            exact = exact + trace_path_distance(
                paths[item.correct_index],
                paths[item.peer_index],
            )
            exact = exact - trace_path_distance(
                paths[item.correct_index],
                paths[item.wrong_index],
            )
        exact = exact / len(triplets)
        exact_gradient = torch.autograd.grad(exact, paths, retain_graph=True)[0]

        surrogate = local_ranking_surrogate(
            paths,
            paths.detach(),
            triplets,
        )
        surrogate_gradient = torch.autograd.grad(surrogate, paths)[0]
        self.assertTrue(
            torch.allclose(exact_gradient, surrogate_gradient, atol=1e-5, rtol=1e-4)
        )


class GradientGuardTests(unittest.TestCase):
    def test_conflict_projection_and_norm_cap(self):
        task = [torch.tensor([1.0, 0.0])]
        ranking = [torch.tensor([-2.0, 2.0])]
        merged, metrics = merge_accuracy_guarded_gradients(
            task,
            ranking,
            max_ratio=0.25,
        )
        guarded_ranking = merged[0] - task[0]
        self.assertEqual(float(metrics["conflict"].item()), 1.0)
        self.assertGreaterEqual(float(torch.dot(guarded_ranking, task[0]).item()), -1e-6)
        self.assertLessEqual(
            float(guarded_ranking.norm().item()),
            0.25 * float(task[0].norm().item()) + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()

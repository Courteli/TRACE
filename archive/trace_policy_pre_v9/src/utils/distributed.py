"""Distributed runtime helpers for rank-isolated local GPU visibility."""

from lightning.fabric.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy


class RankIsolatedTorchElasticEnvironment(TorchElasticEnvironment):
    """Allow four external ranks that each expose one local CUDA device."""

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if num_devices != 1:
            raise ValueError(
                "Rank-isolated TRACE DDP requires exactly one visible GPU "
                f"per process, got devices={num_devices}"
            )
        if self.world_size() != 4:
            raise ValueError(
                "Formal TRACE DDP requires WORLD_SIZE=4, got "
                f"{self.world_size()}"
            )


class RankIsolatedDDPStrategy(DDPStrategy):
    """Use torchrun's global world/rank for distributed data sharding."""

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        return {
            "num_replicas": self.world_size,
            "rank": self.global_rank,
        }

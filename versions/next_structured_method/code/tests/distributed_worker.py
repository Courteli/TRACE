"""Subprocess-only two-rank CPU/Gloo smoke test; never touches CUDA."""
from pathlib import Path
import sys
import torch
import torch.distributed as dist

from fixtures import examples, tiny_config, tiny_model
from trace_structured.data import Example
from trace_structured.runner import fit
from trace_structured.training import reduce_gradients


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Parameter(torch.tensor(2.))
        self.unused = torch.nn.Parameter(torch.tensor(3.))

    @property
    def device(self):
        return self.used.device


if __name__ == '__main__':
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        rank = dist.get_rank()
        for count in (1, 2):
            probe = Probe()
            local = 1 if count == 2 or rank == 0 else 0
            if local:
                (probe.used * (rank + 1)).backward()
            assert reduce_gradients(probe, local) == count
            assert probe.used.grad == (1.5 if count == 2 else 1.)
            assert probe.unused.grad is None
        config = tiny_config(global_batch_size=2)
        train = [*examples(), Example('train:2', 'One plus two?', ('1+2=3',), '3')]
        fit(tiny_model(config, lora=True), train, examples('val'), Path(sys.argv[1]) / 'parallel',
            'stage0', {'fixture': 'distributed'})
    finally:
        dist.destroy_process_group()

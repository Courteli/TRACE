"""Restricted loading for trusted-format Lightning/OmegaConf checkpoints.

PyTorch's weights-only unpickler rejects OmegaConf metadata by default.  TRACE
checkpoints contain only tensors plus OmegaConf containers, so we allowlist the
specific data-container classes needed to decode that metadata while retaining
``weights_only=True``.  No project model/function globals are permitted.
"""

from collections import OrderedDict, defaultdict
from typing import Any

import omegaconf.nodes as omegaconf_nodes
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata


_OMEGACONF_NODE_TYPES = tuple(
    getattr(omegaconf_nodes, name)
    for name in dir(omegaconf_nodes)
    if name.endswith("Node")
)

_SAFE_LIGHTNING_GLOBALS = (
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Metadata,
    Any,
    dict,
    list,
    tuple,
    set,
    defaultdict,
    OrderedDict,
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
    *_OMEGACONF_NODE_TYPES,
)

def install_safe_checkpoint_globals() -> None:
    """Idempotently install the narrow data-only checkpoint allowlist.

    Lightning performs its own ``torch.load`` when resuming optimizer and
    scheduler state.  These globals therefore must remain registered for the
    lifetime of the process.  Do not wrap the same objects in
    ``torch.serialization.safe_globals``: that context manager removes them on
    exit even when they were already registered globally.
    """

    torch.serialization.add_safe_globals(_SAFE_LIGHTNING_GLOBALS)


# Install at import time for Lightning's internal resume path.
install_safe_checkpoint_globals()


def safe_load_checkpoint(path, *, map_location="cpu"):
    """Load a Lightning checkpoint without enabling arbitrary pickle code."""
    # Reinstall defensively in case another library cleared PyTorch's global
    # allowlist after this module was imported.  ``add_safe_globals`` is set-
    # based and safe to call repeatedly.
    install_safe_checkpoint_globals()
    return torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

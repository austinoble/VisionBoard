"""
Adapter registry. To plug in a new model:

  1. copy the template in docs/ADAPTERS.md into app/adapters/<your_model>.py
  2. import it below and add an instance to ADAPTERS
  3. restart the GUI — it appears in the "Run inference" model dropdown

Nothing else in the GUI needs to change.
"""
from .base import ModelAdapter
from .dino_ensemble import DinoEnsembleAdapter

_all = [
    DinoEnsembleAdapter(),
]

ADAPTERS: dict[str, ModelAdapter] = {a.name: a for a in _all}

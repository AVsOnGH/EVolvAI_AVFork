"""
EVolvAI Generative Counterfactual Framework package.

Imports of torch-dependent modules (models, data_loader, train, generate)
are intentionally lazy so that numpy-only commands (e.g. `python run.py mock`)
work on machines without PyTorch installed.
"""

from . import config

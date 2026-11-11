"""EnergyWAM: Latent Transition Energy World Action Model.

A trainable scalar energy model over (condition, candidate future latent,
candidate action), with full-unroll energy-gradient refinement for
Future->Action, Action->Future and joint Action-Future generation.

See README.md for the Chinese implementation notes and IMPLEMENTATION_STATUS.md
for what has actually been run.
"""

__version__ = "0.1.0"

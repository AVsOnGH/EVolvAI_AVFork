"""
Centralized configuration for the EVolvAI Generative Counterfactual Framework.
All hyperparameters, paths, and scenario definitions live here.
"""

# ─── Grid Topology ───────────────────────────────────────────────────────────
NUM_NODES       = 50        # Number of nodes in the grid
SEQ_LEN         = 24        # Hours in a day (temporal resolution)
NUM_FEATURES    = NUM_NODES + 1  # Node demand channels + 1 weather channel

# ─── Dataset ─────────────────────────────────────────────────────────────────
NUM_SAMPLES     = 1000      # Number of simulated daily profiles
BATCH_SIZE      = 32

# ─── Model Architecture ─────────────────────────────────────────────────────
TCN_CHANNELS    = [32, 64]  # Channel sizes for each TCN layer
KERNEL_SIZE     = 2
DROPOUT         = 0.2
LATENT_DIM      = 16        # Dimensionality of the VAE latent space Z
COND_DIM        = 2         # Dimensionality of the condition vector C
DECODER_HIDDEN  = 128       # Hidden layer size in decoder FC

# ─── Training ────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-3
EPOCHS          = 10
KLD_WEIGHT      = 1.0       # Weight for KL divergence term in loss

# ─── Baseline Condition (used during training) ──────────────────────────────
# [Extreme_Weather_Flag, Electrification_Multiplier]
BASELINE_CONDITION = [0.0, 1.0]

# ─── Counterfactual Scenarios ────────────────────────────────────────────────
SCENARIOS = {
    "extreme_winter_storm": {
        "description": "Extreme winter storm + 2.5x fleet electrification surge",
        "condition": [1.0, 2.5],
    },
    "summer_peak": {
        "description": "High summer temperatures + 1.5x electrification",
        "condition": [0.5, 1.5],
    },
    "full_electrification": {
        "description": "Normal weather + 3.0x full fleet electrification",
        "condition": [0.0, 3.0],
    },
}

# ─── Output Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR          = "output"
MOCK_TENSOR_PATH    = f"{OUTPUT_DIR}/mock_demand_tensor.npy"
MODEL_SAVE_PATH     = f"{OUTPUT_DIR}/gcvae_model.pt"

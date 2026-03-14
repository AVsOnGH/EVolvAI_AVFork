# EVolvAI: Data Handoff & Team Expectations

This document outlines the specific data exchange protocols between the **Generative AI Modeler**, **Grid Physics (Akshay)**, and **Optimization (Teammate B)**.

---

## 🏗️ Data Flow Overview

1.  **AI Modeler** → **Grid Physics**: Provides 24-hour demand profiles (kW) per node.
2.  **Grid Physics** → **Optimization**: Provides grid penalty metrics (voltage drops, thermal overloads) based on those profiles.
3.  **Optimization** → **AI Modeler**: (Future) Feedback on critical node bottleneck configurations to refine counterfactual scenarios.

---

## 📤 AI Modeler Output (What you GIVE)

The AI model provides the temporal demand distribution.

| Spec | Value | Details |
| :--- | :--- | :--- |
| **File Format** | `.npy` (NumPy Array) | Efficient binary format for large tensors. |
| **Shape** | `[24, N]` | `24` rows (hourly) by `N` columns (grid nodes). |
| **Default N** | 50 | Configurable in `generative_model/config.py`. |
| **Unit** | kW | Active power demand per node. |
| **Location** | `output/` | e.g., `output/extreme_winter_storm.npy`. |

### Usage for Akshay (Grid Physics):
```python
import numpy as np
demand_matrix = np.load('output/extreme_winter_storm.npy')
# demand_matrix[hour_index, node_index] -> kW value
```

---

## 📥 AI Modeler Input (What you TAKE)

To train high-accuracy models, the AI Modeler requires historical data from the team or external sources.

### 1. Grid Topology (From Akshay)
*   **Expectation:** A list or mapping of nodes to their physical characteristics (transformer capacity, residential vs. commercial type).
*   **Format:** `.json` or `.csv`.
*   **Purpose:** Allows the AI to cluster nodes and generate type-specific demand distributions.

### 2. Historical Charging Logs (From External/Team)
*   **Expectation:** Raw timestamped charging events (Start Time, Stop Time, Energy Delivered in kWh, Peak Power in kW).
*   **Format:** `.csv` or `.parquet`.
*   **Purpose:** The "Ground Truth" used to train the VAE baseline behavior.

### 3. Exogenous Variables (Weather/Traffic)
*   **Expectation:** Temp, Precipitation, or Traffic volume for the same time periods as the charging logs.
*   **Purpose:** Feeds the "Condition Vector" in the VAE.

---

## 🔄 The "Async Handoff" Protocol

To prevent pipeline blocks, we follow this sequence:

1.  **Phase A (Immediate):** AI Modeler provides `output/mock_demand_tensor.npy`.
    *   *Teammates:* Build your simulation/optimization code against this mock file.
2.  **Phase B (Mid-Project):** AI Modeler provides trained counterfactual tensors.
    *   *Teammates:* Replace the mock `.npy` with the real `.npy`. **Your code should require zero logic changes.**
3.  **Phase C (Refinement):** Optimization team identifies "Critical Nodes" that crash the grid.
    *   *AI Modeler:* Adjusts latent triggers to focus demand spikes specifically on those critical nodes for stress-testing.

---

## ⚠️ Data Integrity Rules

*   **Zero Placeholders:** Never hand off arrays containing `NaN` or `None`.
*   **Shape Consistency:** All tensors **MUST** remain `[24, NUM_NODES]`. If the node count `N` changes, it must be updated in `generative_model/config.py` first.
*   **Time Alignment:** Hour `0` is always Midnight-1AM (Standard Time).

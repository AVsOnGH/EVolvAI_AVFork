# 1. Methodology
The main goal of EVolvAI is figuring out what EV charging demand looks like during unprecedented events—like massive winter storms colliding with 100% fleet electrification. Traditional forecasting models just predict the future based on the past, but they completely break down when given a scenario that has never happened before in history. To solve this, we are building a generative counterfactual framework using a conditional Variational Autoencoder (cVAE) paired with a Temporal Convolutional Network (TCN).

Here is how the architecture actually generates a 24-hour demand profile across 50 simulated grid nodes:

1. **Temporal Encoding (TCN Encoder):** We start by taking historical charging demand sequences and running them through a series of dilated 1D convolutions. This structure is fantastic for capturing the full 24-hour cycle of human charging behavior without the vanishing gradient problems you usually see in standard LSTMs.
2. **Latent Representation:** Next, we compress that temporal data into a continuous probability distribution. We are essentially stripping away specific weather occurrences to find the underlying "DNA" or structural shape of normal EV charging.
3. **Causal Intervention:** Now comes the fun part where we ask "what if?". We build a 5-dimensional Condition Vector that holds specific, manipulated variables—like abnormal temperature drops or a 2.5x electrification multiplier. We take this vector and force-feed it into our latent representation to actively change the scenario.
4. **Synthesizing the Counterfactual (TCN Decoder):** Finally, we decode that merged representation back into the real world. The decoder spits out a brand new, physically realistic 24-hour demand curve that mathematically lines up with our hypothetical extreme scenario.

---

# 2. Dataset and Experimentation Plan

To successfully train this model and prove we aren't just generating statistical noise, we need a solid mix of human behavior logs, environmental data, and hard grid constraints. 

## Expected Dataset Requirements

To successfully train this model, we are seeking high-resolution datasets that bridge human mobility behavior, environmental factors, and electrical grid constraints. Ideally, our training pipeline requires three distinct data streams that can be temporally aligned:

* **EV Charging Session Logs:** We need historical, granular charging session data. Ideally, this would include exact timestamps for when a vehicle plugged in and unplugged, total energy delivered (in kWh), and peak power delivery limits. The more geographically diverse, the better.
* **Granular Weather Data:** We need historical weather patterns that align geographically and temporally with the charging logs. We are specifically looking for temperature anomalies and solar irradiance. These variables are critical for defining our conditioning vector.
* **Grid Topology/Load Constraints:** To ground our outputs in physical reality, we need topological data for a representative distribution grid (like a standard 50-node electrical grid). This should ideally track nodal baseline loads and local transformer kVA maximum thresholds.

## Validation and Experimentation Plan

We evaluate our model not just on whether it can predict tomorrow, but whether it acts rationally during extreme "stress tests." Our validation happens in three primary phases:

1. **Reconstruction Fidelity:** Before we try anything crazy, the model has to prove it understands normal, everyday life. We train it to recreate standard historical days on a withheld test set, balancing Reconstruction Loss against KL divergence to ensure the curves actually look realistic.
2. **Counterfactual Stress-Testing:** Once the baseline works, we intentionally break the mold. We take a normal latent day and manually override the condition vector—say, telling the model the temperature just dropped 15 degrees and the EV fleet doubled. We then check if the total kW demand spikes correctly and if the targeted charging peaks physically shift (like drivers charging way earlier in the day due to cold weather battery drain).
3. **Downstream Grid Validation:** The ultimate proof of work is plugging our generated data into a real grid simulator. We pass our synthetic `.npy` tensors directly into an alternating current (AC) power-flow engine to make sure our generative numbers don't crash the physical math or cause impossible nodal violations.

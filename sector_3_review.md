# Sector 3: Generative Counterfactual Framework for EV Demand Modeling

## 3.1 Introduction and Theoretical Motivation
The integration of Electric Vehicles (EVs) into the existing power grid introduces unprecedented stochasticity and magnitude into demand profiles. Traditional deterministic forecasting models are insufficient for modern grid planning, particularly when analyzing "what-if" scenarios crucial for resilience and optimization. This section reviews the current state of deep learning applied to power systems and justifies the necessity of our proposed Generative Counterfactual Framework.

## 3.2 Review of Current Literature

Current literature in EV demand forecasting predominantly relies on standard time-series architectures such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks mapping historical data to future predictions. While effective for deterministic baseline forecasting, these models fail to provide actionable insights for grid topology planning under extreme or unseen conditions.

Recent advancements in generative AI, specifically Variational Autoencoders (VAEs) and Diffusion Models, have shown promise in synthesizing high-fidelity tabular and time-series data. However, their application in power systems is often limited to unconditional generation or simple conditional forecasting (e.g., conditioning on a day of the week).

Furthermore, the integration of Causal Machine Learning in energy systems is primarily focused on interpretable models rather than high-dimensional generative modeling. 

## 3.3 The Gap: Lack of Intervention-Based Latent Conditioning

The critical gap in existing literature is the lack of **"intervention-based latent conditioning"**. Most models observe and map normal, deterministic behaviors. They cannot answer counterfactual questions (e.g., "What would the 24-hour demand profile look like if fleet electrification reached 100% during an extreme winter storm?").

Our architecture addresses this by:
1. Utilizing Temporal Convolutional Networks (TCNs) to capture complex local and long-range temporal dependencies without vanishing gradients.
2. Employing a deep VAE to map high-dimensional historical charging and exogenous (weather) variables into a structured latent space.
3. Critically, introducing causal intervention triggers into this latent space, allowing the model to generate counterfactual scenarios that mathematically reflect realistic demand surges physically mapped to the grid topology.

This approach transitions demand modeling from mere observation to active, actionable scenario planning, providing the essential input for downstream grid penalty engines and structural optimization algorithms.

# Scientific Limitations & Honest Research Disclaimers

## 1. Uncertainty of Noise Estimation in Self-Supervised Regimes
- **The Core Ambiguity**: In the absence of ground-truth clean pairs or paired multi-shot exposures, the predicted map $N$ functions as an **empirical uncertainty proxy**, not a physically calibrated sensor noise measurement.
- Real-world sensor noise follows a heteroscedastic Poisson-Gaussian distribution governed by sensor gain (ISO) and quantum efficiency. Without access to raw Bayer metadata or calibrated dark frames, distinguishing fine dark texture from high-frequency sensor noise remains an ill-posed inverse problem.

## 2. Risk of Over-Exposure and Shadow Inversion
- While patch-based exposure losses guide the global luminance toward a natural midpoint (e.g. $E^* \approx 0.55$), scenes containing intentionally dark content (e.g., night sky, deep shadows, black garments) risk artificial brightening.
- LumaGuard-Net mitigates this through bounded residual scaling ($Y + |\alpha| \Delta Y$), but fine-tuning of $\lambda_{\text{exp}}$ is still necessary across diverse domains.

## 3. Masked Blind-Spot Prediction Assumptions
- Masked pixel consistency relies on local spatial auto-correlation: neighboring pixels provide sufficient context to predict the held-out pixel's intensity.
- In extremely fine structures (e.g. thin wires, star points, fine text), blind-spot interpolation can cause mild oversmoothing.

## 4. No-Reference Metric Pitfalls
- Commonly cited no-reference metrics (such as NIQE or entropy) can reward over-sharpened images with amplified noise.
- Low-light research must balance statistical metrics against qualitative visual inspections and full-reference benchmarks where available.

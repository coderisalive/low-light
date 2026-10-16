# Mathematical Formulation of LumaGuard-Net

## 1. Input Representation & Illumination-Noise Estimation
Let $I_{\text{low}} \in \mathbb{R}^{B \times 3 \times H \times W}$ denote the normalized input image where $I_{\text{low}} \in [0, 1]$.
The input luminance $Y \in \mathbb{R}^{B \times 1 \times H \times W}$ is computed using standard BT.601 coefficients:
$$Y = 0.299 R + 0.587 G + 0.114 B$$

Module A (Illumination-Noise Estimator) maps $[I_{\text{low}}, Y] \in \mathbb{R}^{B \times 4 \times H \times W}$ to an estimated spatial illumination map $L$ and a noise uncertainty proxy $N$:
$$L = \text{clamp}(\sigma(\mathcal{D}_L(\mathcal{E}([I_{\text{low}}, Y]))), \epsilon, 1.0)$$
$$N = \sigma(\mathcal{D}_N(\mathcal{E}([I_{\text{low}}, Y])))$$
where $\epsilon = 10^{-4}$ provides a numerical floor preventing zero-division in logarithmic operations.

## 2. Reliability Map & Feature Modulation
The reliability proxy $Q \in \mathbb{R}^{B \times 1 \times H \times W}$ models confidence as proportional to illumination and inversely proportional to noise uncertainty:
$$Q = \sigma\Big(a \cdot \log(L + \epsilon) - b \cdot N + c\Big)$$
where $a, b, c$ are learnable scalars with initial values $a=1.0, b=1.0, c=0.0$.

Given intermediate features $F \in \mathbb{R}^{B \times C \times H \times W}$ extracted by Module B's depthwise-separable residual blocks:
$$G = \sigma\Big(\text{Conv}_{1\times 1}([F, L, N, Q])\Big) \in \mathbb{R}^{B \times C \times H \times W}$$
$$F_{\text{guided}} = F + F \odot G$$

## 3. Bounded Residual Correction
Using invertible color transformations between RGB and YCbCr:
$$Y_{\text{out}} = \text{clamp}(Y + |\alpha| \Delta Y, 0, 1)$$
$$C_{\text{out}} = C + \beta \Delta C$$
$$\Delta Y = \tanh(\text{Head}_{\text{lum}}(F_{\text{guided}}))$$
$$\Delta C = \tanh(\text{Head}_{\text{chrom}}(F_{\text{guided}}))$$
where $\alpha, \beta$ are initial learnable scaling bounds ($\alpha_{\text{init}}=0.2, \beta_{\text{init}}=0.1$).

## 4. Self-Supervised Multi-Term Objective
Optimization is performed without paired ground-truth targets via:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{mask}} \mathcal{L}_{\text{mask}} + \lambda_{\text{exp}} \mathcal{L}_{\text{exp}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{color}} \mathcal{L}_{\text{color}} + \lambda_{\text{struct}} \mathcal{L}_{\text{struct}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}}$$

### Terms:
1. **Masked Blind-Spot Consistency**:
   $$\mathcal{L}_{\text{mask}} = \frac{1}{\sum M} \sum |M \odot (f(I_{\text{masked}}) - I_{\text{low}})|$$
2. **Patch-based Exposure Control**:
   $$\mathcal{L}_{\text{exp}} = \frac{1}{P} \sum_{p=1}^P |\text{AvgPool}_{k\times k}(Y_{\text{out}})_p - E^*|$$
   with target exposure $E^* = 0.55$.
3. **Edge-Aware Illumination Smoothness**:
   $$\mathcal{L}_{\text{smooth}} = \frac{1}{HW}\sum_{i,j} \left(|\nabla_x L| e^{-k |\nabla_x Y|} + |\nabla_y L| e^{-k |\nabla_y Y|}\right)$$
4. **Color Consistency**:
   $$\mathcal{L}_{\text{color}} = 1 - \frac{1}{|\Omega|} \sum_{p \in \Omega} \frac{\bar{I}_{\text{enh}}^{(p)} \cdot \bar{I}_{\text{low}}^{(p)}}{\|\bar{I}_{\text{enh}}^{(p)}\|_2 \|\bar{I}_{\text{low}}^{(p)}\|_2}$$
5. **Structural Consistency**:
   $$\mathcal{L}_{\text{struct}} = 1 - \text{SSIM}(Y_{\text{out}}, Y)$$
6. **Identity Regularization**:
   $$\mathcal{L}_{\text{id}} = \frac{1}{HW}\sum (|\Delta Y| + |\Delta C|)$$

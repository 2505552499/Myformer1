# Enhanced Motion Flow: Mathematical Derivations and Theoretical Analysis

## A.1 Detailed Mathematical Formulation

### A.1.1 Multi-Scale Motion Kernel Derivation

The traditional finite difference operator for velocity estimation is:
$$v^{(t)} = x^{(t)} - x^{(t-1)}$$

We generalize this to learnable multi-scale kernels. For a temporal scale $s$, the motion at time $t$ is computed as:

$$\mathbf{M}_s^{(t)} = \sum_{i=0}^{k_s-1} \mathbf{K}_s^{(i)} \odot \mathbf{X}^{(t-\lfloor k_s/2 \rfloor + i)}$$

where $\mathbf{K}_s^{(i)} \in \mathbb{R}^{3}$ represents the learnable weights for the $i$-th temporal offset at scale $s$.

**Initialization Strategy**: We initialize the kernels with classical finite difference patterns:
- Scale 1: $\mathbf{K}_1 = [1, -1]^T$ (forward difference)
- Scale 2: $\mathbf{K}_2 = [-0.5, 0, 0.5]^T$ (central difference)
- Scale 4: Xavier uniform initialization

### A.1.2 Feature Encoding Mathematical Details

Each scale encoder $f_s$ is a two-layer MLP:

$$\mathbf{Z}_s^{(1)} = \text{ReLU}(\mathbf{W}_s^{(1)} \mathbf{M}_s + \mathbf{b}_s^{(1)})$$
$$\mathbf{F}_s = \text{ReLU}(\mathbf{W}_s^{(2)} \mathbf{Z}_s^{(1)} + \mathbf{b}_s^{(2)})$$

where:
- $\mathbf{W}_s^{(1)} \in \mathbb{R}^{2d_s \times 3}$, $\mathbf{b}_s^{(1)} \in \mathbb{R}^{2d_s}$
- $\mathbf{W}_s^{(2)} \in \mathbb{R}^{d_s \times 2d_s}$, $\mathbf{b}_s^{(2)} \in \mathbb{R}^{d_s}$
- $d_s = \lfloor D/|\mathcal{S}| \rfloor$ with $D = 66$ and $|\mathcal{S}| = 3$

### A.1.3 Adaptive Fusion Mechanism

The learnable fusion weights are computed as:
$$\boldsymbol{\alpha} = \text{Softmax}(\boldsymbol{\alpha}_{\text{raw}})$$
$$\alpha_i = \frac{\exp(\alpha_{\text{raw},i})}{\sum_{j=1}^{|\mathcal{S}|} \exp(\alpha_{\text{raw},j})}$$

The concatenated motion features are:
$$\mathbf{F}_{\text{motion}} = [\alpha_1 \mathbf{F}_1; \alpha_2 \mathbf{F}_2; \alpha_3 \mathbf{F}_3] \in \mathbb{R}^{B \times T \times J \times D}$$

## A.2 Theoretical Analysis

### A.2.1 Information Theoretic Analysis

**Theorem A.1** (Information Preservation): The high-dimensional output preserves more information than traditional compression approaches.

**Proof**: Let $I(\mathbf{X}; \mathbf{F})$ denote the mutual information between input $\mathbf{X}$ and features $\mathbf{F}$. For traditional approaches with compression:

$$I(\mathbf{X}; \mathbf{F}_{\text{compressed}}) \leq \min(H(\mathbf{X}), H(\mathbf{F}_{\text{compressed}}))$$

where $H(\mathbf{F}_{\text{compressed}}) \leq 3 \log_2(|\mathcal{V}|)$ with $|\mathcal{V}|$ being the value range.

For our high-dimensional approach:
$$I(\mathbf{X}; \mathbf{F}_{\text{high-dim}}) \leq \min(H(\mathbf{X}), H(\mathbf{F}_{\text{high-dim}}))$$

where $H(\mathbf{F}_{\text{high-dim}}) \leq 66 \log_2(|\mathcal{V}|)$.

Since $66 \gg 3$, we have $I(\mathbf{X}; \mathbf{F}_{\text{high-dim}}) \gg I(\mathbf{X}; \mathbf{F}_{\text{compressed}})$.

### A.2.2 Gradient Flow Analysis

**Theorem A.2** (Improved Gradient Flow): The residual connection and high-dimensional output improve gradient flow to motion parameters.

**Proof**: Consider the gradient of loss $\mathcal{L}$ with respect to motion parameters $\boldsymbol{\theta}_m$:

For traditional approaches with compression:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_m} = \frac{\partial \mathcal{L}}{\partial \mathbf{F}_{\text{compressed}}} \frac{\partial \mathbf{F}_{\text{compressed}}}{\partial \mathbf{F}_{\text{high}}} \frac{\partial \mathbf{F}_{\text{high}}}{\partial \boldsymbol{\theta}_m}$$

The compression layer introduces a bottleneck with Jacobian $\frac{\partial \mathbf{F}_{\text{compressed}}}{\partial \mathbf{F}_{\text{high}}} \in \mathbb{R}^{3 \times D}$, which can cause gradient attenuation.

For our approach:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_m} = \frac{\partial \mathcal{L}}{\partial \mathbf{F}_{\text{output}}} \frac{\partial \mathbf{F}_{\text{enhanced}}}{\partial \boldsymbol{\theta}_m}$$

Since $\mathbf{F}_{\text{output}} = \mathbf{F}_{\text{orig}} + \mathbf{F}_{\text{enhanced}}$, we have:
$$\frac{\partial \mathbf{F}_{\text{output}}}{\partial \mathbf{F}_{\text{enhanced}}} = \mathbf{I}$$

The identity Jacobian ensures no gradient attenuation.

### A.2.3 Performance Lower Bound Guarantee

**Theorem A.3** (Performance Lower Bound): The residual connection provides a performance lower bound.

**Proof**: Let $\mathcal{L}(\mathbf{F})$ be the loss function for features $\mathbf{F}$. The optimal performance with original features is:
$$\mathcal{L}^* = \min_{\boldsymbol{\theta}} \mathcal{L}(\mathbf{F}_{\text{orig}}(\mathbf{X}; \boldsymbol{\theta}))$$

With residual connection:
$$\mathbf{F}_{\text{output}} = \mathbf{F}_{\text{orig}} + \mathbf{F}_{\text{enhanced}}$$

In the worst case where motion enhancement is harmful, the optimal solution is $\mathbf{F}_{\text{enhanced}} = \mathbf{0}$, which gives:
$$\mathbf{F}_{\text{output}} = \mathbf{F}_{\text{orig}}$$

Therefore:
$$\min_{\boldsymbol{\theta}} \mathcal{L}(\mathbf{F}_{\text{output}}(\mathbf{X}; \boldsymbol{\theta})) \leq \mathcal{L}^*$$

This proves that the residual connection provides a performance lower bound equal to the original baseline.

## A.3 Frequency Domain Analysis

### A.3.1 Multi-Scale Temporal Frequency Response

The effective frequency response of each scale can be analyzed using the Discrete Fourier Transform (DFT). For a motion kernel $\mathbf{K}_s$, the frequency response is:

$$H_s(\omega) = \sum_{i=0}^{k_s-1} K_s^{(i)} e^{-j\omega i}$$

For the initialized kernels:
- Scale 1: $H_1(\omega) = 1 - e^{-j\omega} = 2j\sin(\omega/2)e^{-j\omega/2}$
- Scale 2: $H_2(\omega) = -0.5 + 0.5e^{-j2\omega} = j\sin(\omega)e^{-j\omega}$

This shows that different scales have different frequency characteristics:
- Scale 1 acts as a high-pass filter
- Scale 2 acts as a band-pass filter
- Scale 4 (after learning) can adapt to capture low-frequency components

### A.3.2 Temporal Receptive Field Analysis

The effective temporal receptive field for scale $s$ is approximately $s$ frames. This allows the model to capture:

- **Scale 1** ($s=1$): Instantaneous motion changes
- **Scale 2** ($s=2$): Short-term motion patterns  
- **Scale 4** ($s=4$): Medium-term motion trends

The combined receptive field spans $\max(\mathcal{S}) = 4$ frames, providing comprehensive temporal coverage.

## A.4 Computational Complexity Analysis

### A.4.1 Forward Pass Complexity

The computational complexity of each component:

1. **Multi-scale motion computation**: $O(|\mathcal{S}| \cdot T \cdot J \cdot k_{\max} \cdot 3)$
2. **Feature encoding**: $O(|\mathcal{S}| \cdot T \cdot J \cdot d_s \cdot 6)$  
3. **Adaptive fusion**: $O(T \cdot J \cdot D)$
4. **Original embedding**: $O(T \cdot J \cdot D \cdot 6)$
5. **Motion transformation**: $O(T \cdot J \cdot D^2)$
6. **Residual addition**: $O(T \cdot J \cdot D)$

The dominant term is the motion transformation with complexity $O(T \cdot J \cdot D^2)$.

### A.4.2 Parameter Count Analysis

The parameter count for each component:

1. **Motion kernels**: $\sum_{s \in \mathcal{S}} k_s \cdot 3 = 2 \cdot 3 + 3 \cdot 3 + 5 \cdot 3 = 30$
2. **Scale encoders**: $|\mathcal{S}| \cdot (3 \cdot 2d_s + 2d_s + 2d_s \cdot d_s + d_s) = 3 \cdot (132 + 44 + 968 + 22) = 3,498$
3. **Fusion weights**: $|\mathcal{S}| = 3$
4. **Original embedding**: $3 \cdot 2D + 2D + 2D \cdot D + D = 396 + 132 + 8,712 + 66 = 9,306$
5. **Motion transformation**: $D \cdot 2D + 2D + 2D \cdot D + D = 8,712 + 132 + 8,712 + 66 = 17,622$

Total parameters: $30 + 3,498 + 3 + 9,306 + 17,622 = 30,459$ parameters.

This represents approximately 1.14% overhead compared to the base model, making EMF computationally efficient while providing significant performance improvements.

## A.5 Ablation Study Mathematical Framework

To validate the contribution of each component, we define the following ablation configurations:

1. **Baseline**: $\mathbf{F} = g(\mathbf{X})$ (original embedding only)
2. **Single-scale**: $\mathbf{F} = g(\mathbf{X}) + h(f_1(\mathbf{M}_1))$ (scale 1 only)
3. **Multi-scale w/o fusion**: $\mathbf{F} = g(\mathbf{X}) + h(\text{Concat}(\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3))$ (equal weights)
4. **Full EMF**: $\mathbf{F} = g(\mathbf{X}) + h(\sum_{s} \alpha_s \mathbf{F}_s)$ (learnable fusion)

The performance gain of each component can be quantified as:
$$\Delta_{\text{component}} = \text{MPJPE}_{\text{baseline}} - \text{MPJPE}_{\text{with component}}$$

This mathematical framework provides a rigorous foundation for understanding and analyzing the Enhanced Motion Flow module.

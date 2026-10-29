# ASCOT distribution generation: model, experiments, and unresolved errors

**Status as of 29 October 2026.** This report describes the m5300/G1600 ASCOT distribution model. It does not describe NSTX-Dreamer. Results below are development-validation measurements, not a demonstrated physics surrogate or a production release.

## Executive summary

The model generates all ten recorded time frames jointly, conditioned on equilibrium geometry and pressure/profile information. It represents each sampled cell's momentum distribution with an explicit log-amplitude and a 16-component shape code, compresses the spatial field, and uses conditional flow matching to generate a complete compressed trajectory. A frozen spatial decoder and momentum decoder turn that trajectory back into nonnegative histogram-bin masses.

This hierarchy is computationally practical, but amplitude accuracy has been its central failure. The spatial autoencoder can produce large amplitude errors even when given the encoded **true** distribution. Generated trajectories can produce still larger, highly localized errors. The inverse amplitude transform magnifies errors: one unit in the standardized amplitude channel corresponds to approximately a factor of **71,430** at large amplitudes. Small average latent losses therefore do not establish useful physical accuracy.

Three successive interventions have measured benefits:

1. Differentiating the amplitude penalty through the actual 16-step generation path improved generation substantially over a detached-prefix, Euler-endpoint approximation. In paired repeat job 5454, pooled four-draw ensemble amplitude relative L1 fell from **152,096.23 to 87.2251**. That remained an absolute failure.
2. Calibrating only the spatial decoder's **865 amplitude-output parameters** on reconstructed training latents, job 7711, reduced the same ensemble metric to **0.37135**. The worst individual draw still had relative L1 **15.3358**.
3. Job 7834 compared a further reconstruction-only calibration against an equal mixture of reconstructed and generated training latents. The mixture gave the best measured pooled generated-amplitude result in this comparison: ensemble relative L1 **0.33857**, individual-draw pooled relative L1 **0.42040**, and worst draw **5.13922**. Reconstruction-only continuation had worst draw **11.67059**.

**Neither 7711 nor 7834 is promoted.** Each calibration recipe passed 10 of 11 absolute gates and failed the required worst-draw relative L1 below 2. Two generated validation draws remain dominated by individual spatial/time-cell spikes. Calibration changes physical amplitude, not the frozen shape channels; it does not solve the known momentum-shape reconstruction limitations or establish conditional diversity, transport fidelity, or uncertainty calibration.

The 225 held-out test records remain excluded from fitting, model selection, and the recent scientific evaluations. This should not be described as “their files have never been opened”: earlier cohort provenance checks and archival preparation included cached files from all splits. No test scientific results are used in this report. All new work for this report was inspection and documentation; no training or evaluation allocation was launched.

## 1. Scientific target and what a frame means

The target is the recorded ASCOT `distrho5d` histogram, with axes

$$
(\rho,\theta,\phi,p_\parallel,p_\perp,t,\mathrm{charge}).
$$

One charge species is used. A native frame has $102\times41\times41\times61\times51=533{,}418{,}282$ bins, about 2.13 GB as float32. The trajectory representation retains every eighth spatial bin in each direction, leaving **13 × 6 × 6 = 468 sampled spatial cells**, all **61 × 51 = 3,111 momentum bins**, and ten physical-time bins. This is sampling, not pooling over omitted bins or a dense reconstruction of them.

| Coordinate | Recorded extent and selection |
|---|---|
| $\rho$ | Native edges 0 to 1; selected indices 0, 8, …, 96 |
| $\theta$ | Native edges numerically 0 to 6.283185307179587 **degrees**; selected indices 0, 8, …, 40 |
| $\phi$ | Native edges numerically 0 to 3.1415926535897936 **degrees**; selected indices 0, 8, …, 40 |
| $p_\parallel$ | 61 bins, approximately −8.80353616×10⁻²⁰ to +8.80353616×10⁻²⁰ kg m s⁻¹ |
| $p_\perp$ | 51 bins, 0 to approximately 8.80353616×10⁻²⁰ kg m s⁻¹ |
| Time | Ten 0.02 s bins spanning 0 to 0.2 s; centers 0.01, 0.03, …, 0.19 s |

The angular numbers look like radians, but the recorded units are **degrees**. Historical wrapper inputs supplied radian-looking extents to degree-valued ASCOT options. The audited ASCOT conversion path and stored metadata agree on degrees. Existing samples must not be relabeled as full poloidal/toroidal coverage. The current network uses finite boundaries on this recorded sector; it does not infer missing angles. Earlier periodic models were gated by this issue. Corrected-input preparation exists, but this report does not claim a completed replacement ASCOT campaign.

Let $H$ denote accumulated marker weight times residence time in a histogram bin. The modeled quantity is $h=H/\Delta t$, with recorded marker weights in s⁻¹. It is a sampled weighted occupancy/rate proxy, **not a whole-device particle count, density in m⁻³, or fast-ion loss fraction**. Summing all ten time-bin amplitudes is the metric normalization used here, not a claim that distinct time bins contain distinct particles. A sustained-source response requires separately justified source-history integration; it is not implicit in these frames. The source-integration implementation is described in [the distribution notes](../distribution_latent.md) and [source integration code](../../alpha_analysis/ai/integrate_distribution_source.py).

Native histogram entries are bin masses. Their total is a direct bin sum. When a plotted quantity is expressed per momentum-coordinate width, integration uses $\Delta p_\parallel\Delta p_\perp$. There is no additional inferred $2\pi p_\perp$ or spatial Jacobian. Applying one would change the target definition. See the [ASCOT postprocessing documentation](https://ascot4fusion.github.io/ascot5/postproc.html) for the distinction between histograms and derived distributions; local recorded metadata governs these particular files.

Zero target cells are real observations on the sampled grid. They remain in spatial amplitude losses and diagnostics. Unmeasured spatial locations are not zeros. Momentum shape training masks cells with zero total amplitude because their normalized shape is undefined; it does not discard amplitude errors at empty cells. White gaps in delivered spatial plots denote unsampled locations.

## 2. Dataset, splits, and conditioning

The cohort audit considered 1,514 candidate run folders and retained **1,502 valid runs representing 1,501 unique geometries**, in one compatible grid/schema/protocol bucket. Four truncated runs and eight permission-denied runs were excluded. All retained records passed the required equilibrium/pressure and AFSI-readiness checks. Geometry-grouped seed-0 partitioning produced:

| Partition | Runs | Unique geometries | Role |
|---|---:|---:|---|
| Training | 1,051 | 1,051 | Parameter fitting, normalization, penalty-gradient calibration |
| Development validation | 226 | 225 | Selection, diagnostics, comparison gates |
| Held-out test | 225 | 225 | Reserved scientific evaluation; not used for these decisions |

The repeated geometry remains within one partition. Repeated use of the development set, including hotspot-driven hypotheses, means these are development results; they are not an unbiased final generalization estimate. Older 13/3 momentum-pilot and scalar-loss splits are separate experiments and must not be conflated with this split.

The current generator is the `eq_pressure` route, with **6,781 static conditioning features**: 6,690 geometry features, 17 DESC pressure values in Pa on $\rho=0,1/16,\ldots,1$, four kinetic/profile groups with a presence flag and 17 samples each (electron density, electron temperature, ion temperature, atomic number), and two symmetry presence/value features. Density is in m⁻³, temperatures in eV, and atomic number is dimensionless. Missing-profile information is represented explicitly. The geometry vector concatenates DESC coefficients `_R_lmn`, `_Z_lmn`, available `_L_lmn`, available flux `_Psi`, and the scalar resolution/period values `_L`, `_M`, `_N`, `_NFP`, in that order. Basis metadata and array shapes are checked through the conditioning schema. These are global equilibrium features, not a magnetic-field tensor supplied at each generated cell. Simulation-physics settings are audited as separate metadata and held within the compatible protocol bucket; they are not automatically conditioning inputs.

An earlier `eq_pressure_afsi` route additionally used a 65-feature source summary: log1p(total source rate), a 16-bin radial marginal, a 32-bin energy marginal spanning 0–10⁷ eV, and a 16-bin pitch marginal spanning −1 to 1. Marginals are normalized and conservatively rebinned by overlap. Its separate 65→32→32 MLP avoids broadcasting thousands of source bins. **The current 5454/7711/7834 model has AFSI dimension zero.** The earlier source-conditioned route failed amplitude accuracy too; it is not evidence that AFSI can never help.

## 3. Architecture and transforms

### 3.1 Momentum representation: amplitude plus shape

For the 3,111 bin masses $h_j$ at a cell/time,

$$
A=\sum_jh_j,\qquad p_j=h_j/A\;(A>0),\qquad a=\log(1+A/s),\quad s=1.
$$

The shared momentum encoder receives $\log(1+3111p)$. It is an MLP 3111→256→256 with SiLU activations, followed by 16-dimensional mean and log-variance heads. The current deterministic AE uses the mean; the weak-VAE comparison uses sampled shape latents, with log variance clamped to [−12,8]. Empty cells have zero amplitude and zero shape code.

Each local representation has **17 channels**: $[a,z_{\rm shape,1:16}]$. The decoder is 16→256→256→3111 with SiLU hidden activations and a final softmax $q$. Reconstruction is

$$
\widehat h_j=\widehat A q_j,\qquad
\widehat A=s\,\operatorname{expm1}(\max(a,0)).
$$

The explicit amplitude is lossless for the **momentum AE's own encode/decode path**, up to numerical precision. It is not lossless after spatial compression or conditional generation. Nonnegative spectra and normalized shape do not enforce correct amplitude.

The momentum loss, averaged over occupied cells, is

$$
L_{\rm mom}=\tfrac12\sum_j(\sqrt{p_j}-\sqrt{q_j})^2
+0.1\,\operatorname{mean}_j[\log(1+3111p_j)-\log(1+3111q_j)]^2
+\operatorname{mean}_{k=1}^3[(p-q)\cdot w_k]^2+\beta\,\mathrm{KL}.
$$

The three moment weights are normalized versions of $p_\parallel,p_\parallel^2,p_\perp^2$. $\beta=0$ for the current AE16 and $10^{-4}$ for the weak-VAE comparison. The KL is the mean diagonal-Gaussian KL per latent component. The full-cohort spectrum cache sampled 16 spatial blocks per run, up to eight contiguous phi cells per block, over all ten times; this is not a cache of only 16 simulations.

Implementation: [SpectrumAutoencoder](../../alpha_analysis/ai/distribution_models.py), [spectrum-cache training](../../alpha_analysis/ai/train_spectrum_cache.py), and [tests](../../tests/test_spectrum_cache.py).

### 3.2 Spatial autoencoder and normalization

The spatial AE treats physical frames independently. Its input is $[B,10,17,13,6,6]$. Each of the 17 channels is standardized using training-only statistics. The encoder uses a replicate-padded 3³ convolution 17→32, SiLU, then a stride-2 3³ convolution 32→16. The result is $[B,10,16,7,3,3]$, or **10,080 compressed values per trajectory**.

Decoding first interpolates to the exact original 13×6×6 sampled grid using trilinear interpolation, `align_corners=False`, then applies 3³ convolutions 16→32→17 with a SiLU between them. All spatial boundaries are finite/replicated; no angular wrap is introduced. The original loss was

$$
L_{\rm spatial}=\operatorname{MSE}(\widehat\ell_{0:17},\ell_{0:17})+
\operatorname{MSE}(\widehat\ell_0,\ell_0),
$$

where $\ell$ denotes standardized local channels. Later spatial pilots added a physical-amplitude penalty while retaining this loss.

Local statistics use 4,918,680 training cell/time samples per channel. Compressed-channel statistics are refit from the selected frozen spatial encoder over 662,130 samples per channel. Means and variances are accumulated in float64 and stored in float32; the standard-deviation floor is $\max(10^{-6}|\mu|,10^{-4})$. Conditioning and AFSI statistics are likewise training-only. The refit of encoded statistics after spatial selection is intentional; the original local normalization is unchanged.

The complete inverse path for a standardized generated endpoint $y$ is

$$
z=y\sigma_{\rm enc}+\mu_{\rm enc},\quad
\ell=D_{\rm spatial}(z)\sigma_{\rm local}+\mu_{\rm local},\quad
\widehat A=\operatorname{expm1}(\max(\ell_0,0)),\quad
\widehat h=\widehat A\,q(\ell_{1:17}).
$$

For amplitude, $\mu_0=19.11248016357422$, $\sigma_0=11.176467895507812$. Thus at large amplitude, an error of one standardized unit multiplies amplitude by $e^{\sigma_0}\approx71{,}429.6$. There is no upper cap. The positivity clamp below zero is part of the existing inverse transform, not a newly added outlier filter.

Implementation: [spatial model](../../alpha_analysis/ai/trajectory_models.py), [normalization/data](../../alpha_analysis/ai/trajectory_data.py), [training and evaluation](../../alpha_analysis/ai/train_distribution_trajectory.py).

### 3.3 Joint conditional trajectory flow

The current `TrajectoryFlow` generates all ten physical frames together. It is **not an autoregressive frame predictor**. Starting from Gaussian noise of shape $[B,10,16,7,3,3]$, it integrates a learned velocity in a separate artificial flow time $\tau\in[0,1]$. Physical time $t$ is conditioning, not the integration variable.

The backbone has width 32. It concatenates three normalized index-coordinate channels to the 16 compressed channels, projects through a 1³ convolution, and adds learned context. Static conditioning is encoded by an MLP 6781→32→32. Physical time, normalized by the maximum recorded center 0.19 s, and flow time each use features $[u,\sin\pi u,\cos\pi u]$ and a 3→32→32 MLP. The optional AFSI MLP is absent from the current route.

A fine-resolution block applies a finite 3³ spatial convolution and a two-sided length-3 temporal convolution with replicated temporal edges. Each residual addition is divided by $\sqrt2$. A stride-2 spatial convolution feeds a coarse block; trilinear upsampling returns to 7×3×3, concatenation with the fine branch gives 64 channels, and a 1³ merge, refinement block, SiLU, and 1³ output predict 16-channel velocity. The current trajectory network contains no attention or transformer. Earlier Transolver work is a distinct precursor.

For standardized encoded target $z^*$, Gaussian $\epsilon$, and uniform $\tau$,

$$
x_\tau=(1-\tau)\epsilon+\tau z^*,\qquad
L_{\rm FM}=\operatorname{mean}\|v_\theta(x_\tau,\tau,c,t)-(z^*-\epsilon)\|^2.
$$

Generation uses 16 fixed Heun steps, 32 velocity evaluations:

$$
k_1=v(y_n,\tau_n),\quad k_2=v(y_n+h k_1,\tau_n+h),\quad
y_{n+1}=y_n+\tfrac h2(k_1+k_2),\quad h=1/16.
$$

This follows the flow-matching formulation of [Lipman et al.](https://arxiv.org/abs/2210.02747), applied in a learned compressed space as in [latent flow matching](https://arxiv.org/abs/2307.08698); the particular architecture and losses above are defined by this repository. The deterministic comparator uses the same backbone with zero input state/flow time and target-latent MSE, without Gaussian sampling. See [trajectory models](../../alpha_analysis/ai/trajectory_models.py), [model tests](../../tests/test_trajectory_models.py), and [pipeline tests](../../tests/test_trajectory_pipeline.py).

## 4. Training recipes and what changed

### Optimizers and selection

The full-cohort momentum AE16 and weak VAE16 trained for 20 epochs, batch 256, hidden width 256. The original spatial AE trained for 30 epochs, two frames per batch, width 32, 16 compressed channels; the normalized-validation-loss selection was epoch 26. The original deterministic and flow routes trained for 30 epochs at one trajectory per update, selecting epochs 23, 30, and 29 respectively for deterministic, geometry/pressure flow, and AFSI flow.

These routes use AdamW with learning rate 0.001, weight decay 0.01, betas (0.9,0.999), epsilon 10⁻⁸, and global gradient clipping at 1. No learning-rate scheduler is present in the audited path. Momentum selection uses validation reconstruction loss; original trajectory selection uses its route's validation training loss. Such selections did not guarantee good decoded amplitudes.

### Physical-amplitude penalties

For record $i$, define $M_i=\sum_{t,c}A_{itc}>0$, and

$$
O_i=\frac{\sum(\widehat A-A)_+}{M_i},\qquad
U_i=\frac{\sum(A-\widehat A)_+}{M_i},\qquad
B_i=\log(1+O_i+U_i).
$$

For the later full-record trajectory and head-calibration recipes, all 468 sampled spatial cells and ten times contribute, including observed zeros. The earlier spatial pilot 3133 is an explicit exception: its penalty numerator covers the current two-frame training minibatch, while its denominator is the full ten-frame record amplitude. Its validation amplitude metrics still cover all ten frames. There is no denominator epsilon. Auxiliary arithmetic uses stable float64 log-domain calculations, avoiding overflow while representing this same objective. The large observed errors are not a division by each individual zero-valued target cell.

The trajectory recipes differ in **where the penalty endpoint comes from**:

| Descriptive method | Endpoint used by amplitude penalty |
|---|---|
| Teacher-interpolation endpoint penalty | $x_\tau=(1-\tau)\epsilon+\tau z^*$, then $y=x_\tau+(1-\tau)v_\theta(x_\tau,\tau,c,t)$ |
| Approximate-output penalty | $x_\tau=\operatorname{stopgrad}(\mathrm{Heun}_{0\to\tau}(\epsilon))$, then the same one-step Euler endpoint $y=x_\tau+(1-\tau)v_\theta(x_\tau,\tau,c,t)$ |
| Full-generation penalty | $y=\mathrm{Heun}_{16}(\epsilon)$; differentiate through all 32 velocity evaluations and the frozen spatial decoder |

In all three, flow matching still uses the teacher-interpolated state in its own MSE term. The original teacher penalty used $\lambda=0.02125897378287374$. The approximate-output and full-generation comparison uses the **same** $\lambda=0.006609611423365146$ in $L=L_{\rm FM}+\lambda B$. Both compared recipes have an amplitude penalty. Historical folder names `control` and `actual_endpoint` are identifiers, not adequate scientific labels.

A separate experiment used $B_{\rm split}=\log(1+O)+\log(1+U)$, with $\lambda=0.005402680000802979$, to reduce the suppression of underprediction gradients when overprediction is huge. This was the unsuccessful separate-over/under arm ending in job 5033. It is not the later full-generation penalty. Penalty coefficients were calibrated on training examples by gradient scale; they are not physically derived constants.

### Amplitude-head calibration

Job 7711 froze the generator, the spatial encoder, the decoder trunk, all 16 shape-output channels, momentum AE, and all normalization. It optimized only the final spatial convolution's amplitude output: $32\times3^3+1=865$ parameters. The objective was $B$ on reconstructed **training** endpoints, with coefficient one and no flow-matching term. Five epochs gave 5,255 updates. Fresh AdamW used learning rate 10⁻⁴, weight decay zero, betas (0.9,0.999), epsilon 10⁻⁸, clip norm 1; training shuffles used seeds 21–25.

Job 7834 began both arms from 7711's calibrated head, with fresh independent optimizers using the same settings and shared training order (seed 26). Each arm received 1,051 updates. One frozen generated endpoint per training record used seed $10^9+\text{manifest index}$. The two objectives were

$$
L_{\rm reconstruction}=B(D(z_i^*),A_i),\qquad
L_{\rm mixture}=\tfrac12B(D(z_i^*),A_i)+\tfrac12B(D(z_i^{\rm gen}),A_i).
$$

Here $D$ includes the amplitude inverse transform, and only its 865 head parameters are trainable. All 904 validation generated endpoints were retained matched inputs, not resampled to favor a result. This tests decoder calibration on the generator's training-domain outputs; it does not improve the generator's latent distribution itself.

## 5. Experiment history

### Earlier representation and data decisions

The earlier Transolver work predicted scalar loss using equilibrium/pressure features and pooled slice tokens. Attempts to generate those learned tokens were a precursor, not a reconstructible distribution generator. Their latent-MSE outcomes cannot establish histogram fidelity. The present hierarchy was introduced to preserve explicit amplitude and decode back to distributions.

The first distribution implementation included a previous-frame-conditioned local/coarse convolutional latent-field flow, with magnetic-field/source context and source/target physical times. Its periodic angular assumptions were blocked by the degree-valued recorded sector. The finite-domain, joint-trajectory model described above supersedes that assumption for this dataset; it does not repair the original recording coverage.

The early momentum study used 16 simulations (13 training, 3 validation), eight sampled blocks per run. Job 59296377 ran a five-epoch screen; job 59296500 ran 50 epochs. The retained comparison contains 1,920 evaluated spectra, 1,492 occupied:

| Method | Shape Hellinger term | Spectrum relative L1 |
|---|---:|---:|
| Training-mean shape with target amplitude | 0.88234 | 1.94695 |
| AE8 | 0.58176 | 1.59388 |
| AE16 | 0.58194 | 1.58903 |
| AE32 | 0.58357 | 1.59440 |
| Weak VAE16, β=10⁻⁴ | 0.56754 | 1.58066 |

These are the receipt's occupied-spectrum shape/reconstruction aggregates, not the later whole-record amplitude metric. AE16 was a compact working representation, not a demonstrated globally optimal latent size. The early weak VAE was slightly better in shape; the full trajectory cache nevertheless used deterministic AE16. Increasing deterministic shape dimension alone did not resolve the error in that pilot.

A related scalar global-loss comparison, job 59296710, used 179 eligible runs (143/36) after censoring and truncation exclusions. It is contextual motivation only, not evaluation of generated distributions. Jobs 59296564 and 59296499 failed before training and were cancelled before starting respectively; they do not count as successful training results. Existing details are in [the October 3 experiment report](../experiment_results_20261003.md).

The full-cohort chain was submitted as audit 59300433, spectrum cache 59300647, and model comparison 59300653; smoke job 59300434 exercised the routes. The retained model-stage summary records successful training and evaluation on October 5–6. It is evidence of completed stages, not by itself a recovered final scheduler accounting receipt for every submitted job. Original resource requests should not be reported as measured GPU consumption.

That first full-cohort evaluation decoded 256 selected spatial cells per record over ten times (578,560 cell/time spectra across 226 records), retaining the full momentum grid. Its reported generated-mean spectrum relative L1 was **1.67708** for deterministic prediction, **2.83505×10¹³** for geometry/pressure flow, and **4.81681×10¹⁰** for AFSI flow. Momentum AE reconstruction alone was **1.51197**; hierarchical reconstruction was **352.88895**. These spectrum metrics are not interchangeable with subsequent full-468-cell amplitude ratios.

Frozen replay job **59425336** reproduced the selected original outputs exactly and exposed omitted spikes outside the 256-cell evaluation subset. It used 109 seconds with four allocated A100s (**436 allocated GPU-seconds**, although only one visible device was used), within its approved two-GPU-hour budget. For G1600_00638, seed 88000, a true-zero cell decoded to approximately 1.006×10³¹; the flow state remained finite and moderate. G1600_00818 failed even in target reconstruction, with approximately 2.940×10¹⁹ predicted at a target amplitude 1.907×10¹⁰. This motivated complete sampled-grid amplitude checks.

### Bounded continuation experiments

The following table distinguishes measured allocation from the experiment-specific cap. A cap is not reusable credit for later jobs. All listed StellarAI computations used one B200. The continuation archive's exact historical directory names are mapped in the reproducibility section.

| Jobs | Change and matched scope | Allocated GPU-seconds / cap | Measured outcome |
|---|---|---|---|
| 3131 / 3133 | Spatial reconstruction: original normalized loss versus added amplitude penalty, λ=0.0030675150498646387; five epochs, 26,275 updates per arm | 171 / 1,800 s, including 1 s failed 3131 | 3131 failed before Python (typed/untyped GRES). At epoch 5, pooled reconstruction error was 100.711 versus 6.25006. The later selected penalty **epoch 1** had error 4.50590; all spatial weights had been trained. |
| 3136 | Freeze selected 3133 epoch-1 spatial AE; refit encoded statistics on training data; train a fresh geometry/pressure flow for five epochs (5,255 updates) | 86 / 900 s | Reconstruction error 4.50590; generated ensemble error 9.34692×10¹³. Finite output and reduced reconstruction error did not make generation usable. |
| 3137 | Frozen replay of G1600_01307, seeds 193000 and 193001; no optimization | 11 / 120 s | Historical metrics reproduced exactly, including draw error 7.96972×10¹⁶. A decoder amplification diagnostic, not a repair. |
| 3142 | From 3136 epoch 5, five matched additional epochs per arm: flow matching alone versus teacher-interpolation endpoint penalty | 156 / 300 s | Ensemble error 5.80480×10¹² versus 1.31975×10¹¹. Improved, but unusable. The saved integration-screen pass is not an accuracy pass. |
| 3275 | From 3142 penalty epoch 10, five matched additional epochs: teacher-interpolation versus approximate-output penalty | 268 / 360 s | Ensemble error 3.40327×10¹¹ versus 1.76854×10⁷. The approximate-output recipe failed underprediction (+2.73048 percentage points), absolute error, and total-amplitude-ratio gates. |
| 5019 / 5023 / 5031 / 5032 / 5033 | From epoch 15, five matched additional epochs: joint over/under penalty versus separate log1p penalties; completed at epoch 20 | 282+34+44+29+77 = **466 s**; earlier 360/416 s caps were superseded by explicit completion authorization; final allocation envelope 600 s | Four allocations were incomplete/failed; 5019, 5031, and 5032 retained progress, while 5023 saved no new updates. Job 5033 completed the last 345 updates. Separate penalty worsened ensemble error: 257,590.99 versus 54,827.40, and underprediction by 1.88314 points. Rejected. |
| 5444 | Epoch 20→21: approximate-output versus full-generation penalty; matched seed 20, 1,051 updates per arm | 170 / 600 s | Relative improvement gates passed; absolute amplitude accuracy failed. |
| 5454 | Independent paired repeat from the same epoch-20 parent; training seed 21, 1,051 updates per arm | 180 / 300 s | Relative improvement reproduced; absolute amplitude accuracy failed. Selected frozen generator for calibration, without production promotion. |
| 7711 | Five-epoch reconstruction-only calibration of 865 amplitude-output parameters; 5,255 updates | 66 / 300 s, csd | 10/11 absolute gates passed; worst draw 15.3358 failed. No promotion. |
| 7834 | From 7711: reconstruction-only versus equal reconstruction/generated-input calibration; 1,051 updates per arm | 70 / 180 s, csd | Each arm passed 10/11 absolute gates; mixture passed 3/3 comparative gates but worst draw 5.13922 still failed. No promotion. |

### Endpoint-gradient comparison: jobs 5444 and 5454

Both jobs initialized each recipe from the same epoch-20 model **and AdamW state**, kept decoder/data/normalization fixed, and added one epoch (1,051 updates per arm). Job 5444 used paired training seed 20; 5454 repeated the experiment with seed 21. Each scored the same 226 development records, four generation draws per record, with the original 16-step sampler and seed $1000\times\text{validation record index}+j$.

| Job | Penalty endpoint | Pooled ensemble relative L1 | Median record ensemble relative L1 | Worst record ensemble relative L1 |
|---|---|---:|---:|---:|
| 5444 | Approximate output | 74150.924 | 16.314934 | 17585693 |
| 5444 | Full generation | 796.63357 | 1.3974536 | 123008.1 |
| 5454 | Approximate output | 152096.23 | 26.548755 | 26307768 |
| 5454 | Full generation | 87.225063 | 1.0265949 | 14627.587 |

Both repeats passed the predeclared relative-improvement gates. The repeat improved 225/226 ensemble records and 897/904 individual draws. Normalized shape MSE changed little. The full-generation recipe's absolute accuracy still failed; large relative gains from an extreme baseline were not a solution. Its frozen generator checkpoint is the common generator used by both later head calibrations.

## 6. Latest matched validation and remaining spikes

Define individual relative amplitude error

$$
R_{ij}=\frac{\sum_{t,c}|\widehat A_{ijtc}-A_{itc}|}{M_i}.
$$

“Pooled” means sum of errors divided by sum of target amplitudes, not the arithmetic mean of record ratios. Individual-draw pooling uses all 904 errors and denominator $4\sum_iM_i$. For the four-draw ensemble, first average **physical amplitudes**, then take L1 against truth. It is not an average of logs, latent states, or individual-draw L1 values. Percentiles and maxima are explicitly per record or per draw.

| Metric | 5454 generator + uncalibrated 3133/3136 decoder | 7711 reconstruction calibration | 7834 reconstruction-only continuation | 7834 reconstructed/generated mixture |
|---|---:|---:|---:|---:|
| Reconstruction pooled relative L1 | 4.5059191 | 0.36859093 | 0.39834369 | 0.38192115 |
| Reconstruction record p95 | 10.977326 | 0.44122941 | 0.49189616 | 0.44015773 |
| Reconstruction worst record | 161.67534 | 4.2710276 | 3.6555423 | 3.2445603 |
| Generated ensemble pooled relative L1 | 87.225063 | 0.37135328 | 0.3740526 | 0.33857303 |
| Generated ensemble record p99 | 390.74518 | 0.63994755 | 0.58709672 | 0.4938933 |
| Generated ensemble worst record | 14627.587 | 4.0732013 | 3.129246 | 1.4917801 |
| Individual-draw pooled relative L1 | 87.354921 | 0.45757934 | 0.4627774 | 0.42040455 |
| Individual-draw median | 0.87037198 | 0.40740572 | 0.436313 | 0.40790028 |
| Individual-draw p99 | 320.25912 | 1.1579954 | 0.878456 | 0.61944776 |
| Individual-draw worst | 58507.924 | 15.335762 | 11.670589 | 5.1392203 |
| Mean individual-draw underprediction | 0.21981702 | 0.23473269 | 0.19765872 | 0.21254639 |
| Median draw total-amplitude ratio | 1.5187322 | 0.98254087 | 1.0801474 | 1.0207832 |

Coverage is exact for **226 target-latent reconstructions and 904 matched generated endpoints**, each over all ten times and all 468 sampled spatial cells. Target amplitudes come from cached local amplitude channels, not newly read raw ASCOT momentum histograms. The original target-latent reconstruction already exceeded relative L1 1 in **189 of 226 records**. This is a broad decoder reconstruction problem, not just two troublesome generated examples.

Job 7711 improved all 226 reconstruction ratios, all 226 ensemble ratios, and all 904 draw ratios against its matched uncalibrated baseline. Its 1,130 reconstructed/generated shape-output comparisons were bitwise unchanged. Job 7834 checked 2,260 such comparisons across both arms; frozen states were unchanged, and the retained baseline GPU log-amplitude replay matched exactly. An independent saved-array verification passed 88 checks. These tests establish what changed and that the saved metrics are reproducible; they do not establish physical validity.

The mixture was not uniformly better than 7711: it improved 121/226 ensemble records and 508/904 individual draws, while slightly worsening pooled reconstruction error. Against matched reconstruction-only continuation it improved 176/226 ensemble records and 783/904 draws. Its lower pooled error and improved upper tail therefore coexist with tradeoffs elsewhere.

### Gates and status

The absolute checks were: reconstruction pooled relative L1 <0.5; reconstruction per-record p95 <1; ensemble pooled relative L1 <1; ensemble per-record p99 <2; **worst individual draw <2**; median draw total-amplitude ratio between 0.5 and 2; mean individual underprediction increase ≤0.02 versus original 5454's 0.2198170227; unchanged shape; unchanged frozen state; finite outputs without overflow or all-zero draws; and complete planned updates/evaluation.

Job 7711 and each 7834 arm passed **10/11**, failing worst draw. The 7834 mixture additionally passed all three comparison checks: lower worst draw than 7711, lower worst draw than reconstruction-only continuation, and pooled ensemble error no worse than 7711. These are development screening thresholds, not demonstrated scientific error tolerances. No production promotion occurred.

### Morphology of the remaining failures

For the 7834 mixture, the two draws exceeding relative L1 2 remain dominated by single spatial/time cells:

| Record and generation seed | Draw relative L1 | Peak generated amplitude | Target at that cell | Fraction of draw error at peak |
|---|---:|---:|---:|---:|
| G1600_01416, 208002 | 5.13922 | 4.45196×10¹⁵ | 3.06798×10¹⁰ | 92.67% |
| G1600_00281, 36001 | 2.56456 | 1.44999×10¹⁵ | 0 | 82.01% |

G1600_01416's peak occurs at time 0.09 s, sampled index [time,rho,theta,phi]=[4,11,4,1], original index [4,88,32,8], approximately (rho,theta,phi)=(0.867647,4.980574°,0.651306°). G1600_00281's peak is at 0.13 s, sampled [6,0,2,0], original [6,0,16,0], approximately (0.004902,2.528599°,0.038312°). One is an occupied target cell and one is an empty target cell. Empty-cell masking alone cannot address both and would hide a real prediction error.

For diagnosis only, excluding each peak would reduce the draw ratios to approximately 0.37664 and 0.46144. Those exclusions are **not** applied to any reported gate or headline metric. Before this comparison, the corresponding 7711 failures each had more than 96% of error in a single cell; the fractions changed after further calibration.

## 7. What the diagnostics establish—and what they do not

**The amplitude decoder is an observed failure site.** Both the original frozen replay and the all-record target reconstruction checks establish failures before conditional generation. The inverse log scale makes occasional decoder overshoots disproportionately costly. Head-only calibration's large improvement supports the practical importance of this path, but it does not prove that the output layer is the sole cause.

**A simple integration or precision defect has not been found.** Saved Heun proposals and corrections reproduced exactly in the original replay; normalization round trips were consistent with float32 precision; momentum softmax decoding conserved its supplied amplitude to approximately 10⁻⁷ relative scale. Later CPU replays retained extreme errors at 16, 64, and 256 steps. For G1600_01093's full-generation endpoint, relative L1 was approximately 1,642.65, 1,740.94, and 1,747.00 respectively; the maximum log-amplitude change from 64 to 256 steps was about 0.00441. The float32/float64 difference at 64 steps was at most about 1.01×10⁻⁴ in log amplitude. These are bounded case studies, not a global convergence proof, and CPU numbers must not silently replace canonical GPU comparisons.

**Marginal latent ranges are insufficient diagnostics.** All 10,080 compressed coordinates of the inspected G1600_01093 generated endpoint lay within the corresponding channel’s training extrema, pooled across training time/space samples, yet decoding failed. Being within those marginal channel ranges does not establish membership in the joint training distribution. Interpolating from target to generated endpoints changed hotspot log amplitude nonlinearly. Decoder sensitivity to joint latent combinations remains a plausible mechanism.

**Normalization and indexing audits found no causal software error in the checked path.** Dataset/conditioning statistics and source/checkpoint hashes matched their receipts. A preparation-shard provenance mismatch was resolved by comparing each cached file to its own shard's metadata, without altering data. Tests covered stable amplitude arithmetic, inverse transforms, masks, shape/momentum conservation, and saved-output recomputation. Finite outputs alone are a weak condition: even the enormous failed predictions were often finite and below float32 overflow.

**Shape remains a separate limitation.** The momentum AE itself has substantial shape reconstruction error. Later shape metrics include standardized 16-channel MSE and probability comparisons decoded from cached target shape codes. The latter are decoder-reference proxies, not direct raw-histogram truth. Some earlier continuation shape evaluations used 256 selected cells and only the first generation draw; they do not establish 904-draw raw-momentum accuracy. Head calibration leaves those shape outputs unchanged by construction.

**The evaluation is not full-device validation.** It covers the retained sparse spatial grid, limited recorded angular sector, fixed time window, and the sampled momentum grid. It has not established correct source-history dependence, general time extrapolation, pressure/transport integrals, calibrated uncertainty, or new-geometry scientific accuracy on the reserved test set.

## 8. Current artifacts and justified next hypotheses

The best measured **generated-amplitude** combination is the 5454 full-generation-penalty generator plus 7834's reconstruction/generated-input calibrated spatial head and the frozen AE16. The 7711 head remains better in pooled target reconstruction; “best” therefore needs a stated metric. Neither is an approved production replacement, and no production-ready stochastic model is established by these measurements. Preserve the original checkpoints and all alternatives.

The next work should test hypotheses rather than add a cosmetic amplitude cap:

1. **Tail sensitivity of the decoder objective.** The log1p record-level penalty compresses very large errors. A training-only tail-aware amplitude objective or a carefully defined cell-tail term could prioritize rare spikes. It would change the science objective and needs a separate controlled proposal, with matched reconstruction-only and generated-input baselines and the same underprediction/shape gates.
2. **Coverage of generated training endpoints.** One generated endpoint per training record may not cover rare latent combinations. More training-only draws or deliberately selected hard training endpoints could test this. Validation hotspots must remain evaluation examples, not be inserted into the training pool.
3. **Representation-level amplitude/occupancy modeling.** A separate amplitude or occupancy representation, improved decoder constraints, or reduced joint-latent sensitivity may be necessary. Each changes architecture or assumptions and must be compared as such. Coordinatewise clipping is not evidence of a valid joint distribution, and masking true-zero locations is not a valid repair.
4. **Conditioning and temporal/spatial adequacy.** Check conditional dependence and generated diversity against simple baselines before concluding that a lower amplitude score is a useful conditional surrogate. Consider AFSI only in an explicit controlled comparison. Correct angular recording coverage requires new data, not relabeling.

No further training, clipping, architecture change, or test-set unsealing is authorized or performed by this report. The immediate deliverable is a documented, reproducible state of the investigation.

## 9. Reproducibility and private artifact locations

This section intentionally records private research locations for the current working group. It requires a disclosure review before public publication. Raw data and checkpoint payloads are not embedded in this report.

### Repository and source identity

The actual worktree is on Perlmutter at `/global/cfs/cdirs/m5300/rchurchi/alpha_analysis_distribution`, branch `codex/full-distribution-latent`. The source baseline HEAD is **`1446ba274ea09b12f447cc0fd173ac06b325f2cc`**, plus a frozen dirty/untracked snapshot. HEAD alone does **not** contain the complete scientific source used for these experiments. `source-manifest.json` records that snapshot's files and hashes. Repo-relative source/test links above resolve in this worktree; some targets remain untracked and will not resolve in a documentation-only GitHub tree. This report does not silently add or alter them.

The user-owned remote is `https://github.com/rmchurch/alpha_analysis.git`; upstream is `https://github.com/pablo-oyola/alpha_analysis.git`. Both were verified public. The distribution work branch had no matching remote head when inspected. This report is prepared as a documentation-only local commit; pushing it would newly disclose experiment results and these internal paths and requires publication authorization. No upstream PR or protected-branch write is part of this task.

### Artifact roots and receipts

| Root label | Private location |
|---|---|
| Original full-cohort runs | Repository `runs/trajectory_full_20261003/` |
| Early momentum comparison | Repository `runs/distribution_comparison_20261003/extended/comparison.json` |
| Original frozen diagnostic | Repository `runs/diagnostic_amplitude_20261006T1210Z/results_59425336/` |
| StellarAI continuation root | `/scratch/gpfs/STELLARATOR/rmc2/m5300/distribution_continuation_20261017_r1` |
| Historical local receipts | `/Users/rchurchi/Documents/Codex/2026-10-16/task-2` |
| Latest diagnosis/calibration evidence | `/Users/rchurchi/Documents/Codex/2026-10-28/task-4/ascot-spike-diagnosis` |

Within the historical receipt root, start with `source-manifest.json`, `reference-dataset-audit.json`, `conditioning-normalization-audit.json`, `shard-provenance-audit.json`, `distribution-status.json`, and `input-transfer-receipt.json`. The raw cohort's roughly 64.7 TB logical contents were not exhaustively hashed; the audit distinguishes raw header/options/size/mtime checks from hashed cached inputs and contexts.

Experiment outcomes are in `pilot-job-3133-outcome.json`, `exploratory-flow-3136-outcome.json`, `frozen-flow-replay-3137-summary.json`, `matched-endpoint-3142-outcome.json`, `generated-state-3275-outcome.json`, `amplitude-completion-5033-outcome.json`, `actual-endpoint-pilot-5444-outcome.json`, and `actual-endpoint-replicate-5454-outcome.json`, with companion accounting, authority, source, and verification files. These are the evidence for the numerical history, including failed attempts.

Under the continuation root, relevant run directories are `runs/spatial_amplitude_pilot_20261017_r1/3133`, `runs/exploratory_flow_20261017_r1/3136`, `runs/split_amplitude_flow_20261023_r1/5033`, and `runs/actual_endpoint_replicate_20261024_r1/5454`. The latest diagnostics are `diagnostics/amplitude-calibration-20261029-r1/7711` and `diagnostics/generated-input-calibration-20261029-r1/7834`.

Locally, `calibration/7711` and `generated-input-calibration/7834` contain `report.json`, `outcome.json`, `accounting.txt`, `paired-validation.h5`, checkpoint files, and artifact/plot manifests. The latter also retains `training-endpoints.h5`, `verify_saved_result.py`, and `result-review.json`. Its scheduler receipt is **COMPLETED, exit 0:0, 2026-10-29 12:20:13–12:21:23 UTC**, account **csd**, one B200, four CPUs, 16 GiB, 70 allocated GPU-seconds against the 180-second hard cap. Job 7711 used the same account/resource shape, 66 GPU-seconds against 300. A prior submission under `stellarator` was rejected before allocation; the user explicitly selected `csd`. Authentication interruptions were not counted as compute runs.

The saved slice plots preserve target/reconstruction/generation distinctions. Earlier momentum panels used frozen CPU endpoint decoding, with measured maximum log differences 0.105/0.082 relative to retained GPU outputs; spatial panels used saved GPU fields. Latest calibration plots derive from retained paired GPU outputs. Plot provenance, masks, degree units, and bin-width weighting must accompany reuse.

### Critical SHA-256 pins

| Artifact | SHA-256 |
|---|---|
| Cohort split | `0aae7139aac1d5f4ad4550e3feced68dfee26ca69d20751e66af3f034dc5233e` |
| Frozen momentum AE16 | `5330b637c3dc75acb2e5d486d050b5ff337724939fab321551bc920051b869be` |
| Selected/refitted spatial AE before calibration | `d9647ca4f9222d1731e98ae5190266c326298c85b07c43b597488c8039ee1f94` |
| Epoch-20 generator/optimizer parent, 5033 joint penalty | `c7fbfd3a592f60bc0640533e928412aab4d09dd0c215a674247d355fe1f43e85` |
| 5454 approximate-output generator | `fcdd01bb0daece388fa1d0c94a8b4443e76cad610736b7070c5f6b4a615c8894` |
| 5454 full-generation generator | `cb9cb3318af9d6987dc8cfa5bed3ab16e28e7677ecbdc385239b45a8ad98dc4d` |
| 7711 calibrated spatial checkpoint | `137117a431c175881d5c95ae4e3f2b92957f105cdbf5ddb6949361286cdefc70` |
| 7834 reconstruction-only spatial checkpoint | `0e851877e938aff9b0ab4eb5b4f470222197c8b4902c1093f2a0b24600f1c856` |
| 7834 reconstruction/generated-input spatial checkpoint | `9112c8cb5bb8d29c8b4037600125355da3eb8f9ef3f1fedcf9616094388b5cb6` |
| 7834 paired validation arrays | `618b3961e080a27c155cc530ada6c181ec84a9130e608cc214206da36fbd1a30` |
| 7834 training endpoints | `8834b71a0b265c5b768d15fc5df7e24a06bd9bc9ff179146347294e85fe1e579` |
| 5454 runner | `6f137a3b07b1047b16e8ec445a8401bd7b1532588f39651b002e03d806c57992` |
| Stable amplitude math helper | `e00a0460a57f35d37b945cf82a32dc2eaa0a7c146049138777df77775a3811e7` |
| 7711 runner | `8db3507b7d5e2460210c943f9b771210dcecf51ef899cff381cbdfebe4ede8b7` |
| 7834 runner | `c0ad24d235aae1df7d1b9835bd22714f2fe1b5b5d76260ec281e99dd91ea4111` |

Earlier selected/comparison artifacts, retained separately from the calibrated alternatives:

| Artifact | SHA-256 |
|---|---|
| Original deterministic trajectory | `90904350eb778799e72810a5364cc240463dab617f22d5b9963f245f9b67032d` |
| Original geometry/pressure flow trajectory | `d6a784bbcd122b58ebb9708d94a04cf5186d5362aeb6908e746020fb28591601` |
| Original AFSI flow trajectory | `e56b3028ed9365bbd9fbb8d4329a15f526e4e629a0495ab6849fed9579ad5519` |
| 3133 spatial amplitude penalty epoch 1 | `2eea5d2ec613c7d46f14e5202226126ffb8d2d1c030b11bedd4ffc7e43226a4a` |
| 3136 fresh flow, epoch 5 | `6b491bdd0ffc5ab8c91562d62db0b49acd9bfd0402c56bc6393df51cef3b32db` |
| 3142 flow matching alone | `867ef2c57478839f03279bf04c73d46d105e27a9ff95b34d571b4e21e5abdc8e` |
| 3142 teacher-interpolation penalty | `6fd2795f5240337a1ec4b1c91e179c217db8645e49f6c3431fe2b2addca0b695` |
| 3275 teacher-interpolation penalty | `8a17651ec052acbcec59519ed04f2317ec47a6417b526b90e7ceb3f8271619cb` |
| 3275 approximate-output penalty | `3c0a9240e0084f3d28d6b5728af706ea66665af1853183df596a7cf1808ce7a8` |
| 5033 separate over/under penalty | `72d5f10a834e12acc31dfa8e7b269b5f8e4fddad26881c225f2da4920599c709` |

To reproduce a comparison, verify the source snapshot and these pins; restore the matching manifest/split and normalization; retain the exact target and generated endpoint identities, sampling seeds, and metric coverage; then run the corresponding saved-array verification. Do not recreate the experiment from HEAD alone, substitute a newly sampled endpoint, or compare a 256-cell spectrum metric to a 468-cell amplitude metric. Full source manifests and outcome receipts carry additional hashes for the earlier experiments.

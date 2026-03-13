## Cost-Aware Mixture of Experts (CA-MoE)
A variation of the Mixture of Experts classifier.
The gating network learns to route samples to cheaper experts when possible, trading off accuracy against computational cost via a tunable `gamma` and `lambda` parameter.

At each forward pass the gating network produces a probability distribution $p(x) \in \Delta^{K-1}$ over $K$ experts.

The training loss is: $\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} p_j(x_i) \left[ \ell\!\left(f_j(x_i),\, y_i\right) + \gamma \cdot c_j \right] - \lambda \cdot \mathcal{H}(p)$

| Symbol | Description |
|---|---|
| $p_j(x_i)$ | Gating probability for expert $j$ on sample $x_i$ |
| $\ell(f_j(x_i), y_i)$ | Cross-entropy of expert $j$'s prediction |
| $c_j$ | Pre-computed cost of expert $j$ (e.g. $\log_{10}(\text{FLOPs})$) |
| $\mathcal{H}(p) = -\frac{1}{N}\sum_i \sum_j p_j(x_i)\log p_j(x_i)$ | Average gating entropy |
| $\gamma \geq 0$ | Cost penalty weight |
| $\lambda \in \mathbb{R}$ | Entropy regularization weight |

---

### Setup

##### 1. Install dependencies

Requires Python 3.10 and [Poetry](https://python-poetry.org/).

```bash
make environment
```

`.venv` stored in project.

##### 2. Configure environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
# CUDA
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
CUDA_MODULE_LOADING=LAZY

# WandB - omit to disable
WANDB_ENTITY=your_wandb_username_or_team
WANDB_API_KEY=your_wandb_api_key
```

WandB is optional. If `WANDB_API_KEY` is not set, logging falls back to CSV only.

---

### Hyperparameters

##### $\gamma$ - accuracy vs. cost

- $\gamma = 0$: standard soft-MoE, no cost awareness
- $\gamma > 0$: the gating network is penalized for routing to expensive experts; higher values increasingly favor cheap experts over accurate ones

##### $\lambda$ - exploration vs. specialization

- $\lambda = 0$: no entropy term
- $\lambda > 0$: maximizes gating entropy - encourages the gating to spread load across experts (exploration)
- $\lambda < 0$: minimizes gating entropy - encourages the gating to collapse to a single expert per sample (specialization)

---

### Running

##### Single run

```bash
ca-moe
```

Key overrides:

| Parameter | Default | Description |
|---|---|---|
| `model` | `moe/lite` | Model config (`moe/lite`, `moe/simple`, `moe/deep`, `expert/nn1`, …) |
| `dataset` | `mnist` | Dataset (`mnist`, `fashion_mnist`, `cifar_100`) |
| `gamma` | `0.0` | Cost penalty weight |
| `lambda_` | `0.0` | Entropy regularization weight |
| `wandb` | `disabled` | WandB config (`enabled`, `disabled`) |

Override other [training parameters](data/conf/training/default.yaml) throught the CLI.

Example:

```bash
ca-moe model=moe/deep dataset=cifar gamma=0.1 lambda_=0.05 epochs=10 patience=5 lr=0.01
```
Resume from the last checkpoint of a previous run:
```bash
ca-moe model=moe/deep dataset=cifar gamma=0.1 lambda_=0.05 epochs=25 patience=10 lr=0.001 resume=true
```
##### Parallel sweep

Uses [hydra-joblib-launcher](https://hydra.cc/docs/plugins/joblib_launcher/) to run jobs in parallel across all CPU cores.

```bash
ca-moe -m hydra=sweep +gamma=0.0,0.05,0.1,0.5
```

Comma-separated values produce one job per value. Multiple parameters produce a grid:

```bash
ca-moe -m hydra=sweep +gamma=0.0,0.1 +lambda_=0.0,0.1
```

Logs written to `data/experiments/<dataset>/<model>/<run_name>/`.

---

### Structure

```
ca_moe/
  models/         # Expert architectures, MoE model, ExpertWrapper
  factories/      # Trainer setup, cost functions
  datasets/       # LightningDataModules (MNIST, FashionMNIST, CIFAR100)
  utils/          # Logging, tools, WandB config
data/
  conf/           # Hydra configs (model, dataset, training, hydra)
  datasets/       # Downloaded datasets
  experiments/    # Training outputs (checkpoints, CSV logs, WandB logs)
tests/            # pytest test suite
```

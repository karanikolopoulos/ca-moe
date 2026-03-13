from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from ca_moe.utils.tools import eval_mode, build_optimizer


@dataclass
class MoEMetrics:
    probas: torch.Tensor  # (batch_size, num_experts) - gating probabilities

    # __post_init__
    gating_entropy: torch.Tensor = field(init=False)
    expert_probs_mean: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Gating entropy: -sum(p * log(p)) averaged over batch
        self.gating_entropy = (
            -(self.probas * torch.log(self.probas + 1e-10)).sum(dim=1).mean()
        )

        # Mean probability for each expert across batch
        self.expert_probs_mean = self.probas.mean(dim=0)


class CostAwareMoE(L.LightningModule):
    """
    Cost-Aware Mixture of Experts with PyTorch Lightning

    Args:
        experts: Dictionary or list of expert models
        gating: Gating network configuration or instance
        gamma: Weight for cost in loss function (accuracy vs cost trade-off)
        lambda_: Weight for entropy regularization (encourage exploration/specialization)
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam', 'sgd', etc.)
    """

    def __init__(
        self,
        experts: dict[str, nn.Module],
        gating: nn.Module,
        cost_fn: callable,
        gamma: float,
        lambda_: float,
        learning_rate: float,
        optimizer: str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["experts", "gating", "cost_fn"])

        self.experts = nn.ModuleDict(experts)
        self.gating = gating

        self._compute_expert_costs(cost_fn)

        self.expert_names = list(self.experts.keys())

    def _compute_expert_costs(self, cost_fn: callable) -> None:
        experts = list(self.experts.values())
        costs, raw_costs = [], []
        for e in experts:
            with eval_mode(e):
                costs.append(cost_fn(e))
                raw_costs.append(cost_fn.raw(e))
        self.register_buffer("costs", torch.tensor(costs, dtype=torch.float32))
        self.register_buffer("raw_costs", torch.tensor(raw_costs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through MoE

        Args:
            x: Input tensor (batch_size, *input_dims)

        Returns:
            Tuple of (gating_probabilities, expert_outputs)
            - gating_probabilities: (batch_size, num_experts)
            - expert_outputs: list of (batch_size, num_classes) tensors
        """
        # Get expert predictions
        outputs = [expert(x) for expert in self.experts.values()]

        # Get gating probabilities
        logits = self.gating(x)
        probas = F.softmax(logits, dim=1)

        return probas, outputs

    def _compute_loss(
        self, probas: torch.Tensor, outputs: list[torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, MoEMetrics]:
        """
        Compute cost-aware loss

        Args:
            probas: Gating probabilities (batch_size, num_experts)
            outputs: Expert outputs, list of (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)

        Returns:
            Tuple of (total_loss, metrics)
        """
        batch_size = labels.size(0)

        # Compute cross-entropy loss for each expert (no reduction)
        # Shape: (num_experts, batch_size)
        experts_loss = torch.stack(
            [F.cross_entropy(out, labels, reduction="none") for out in outputs]
        )

        # Expand costs to match batch: (num_experts, batch_size)
        costs_expanded = self.costs.unsqueeze(1).repeat(1, batch_size)

        metrics = MoEMetrics(
            probas=probas,
        )
        # Cost-aware loss: weighted sum over experts, mean over batch
        # probas.T: (num_experts, batch_size)
        # Formula: E_sample[ sum_expert(p_j * (CE_j + gamma * cost_j)) ]
        loss = (
            probas.T * (experts_loss + self.hparams.gamma * costs_expanded)
        ).sum() / batch_size

        # encourage exploration (lambda_ > 0) or specialization (lambda_ < 0)
        loss -= self.hparams.lambda_ * metrics.gating_entropy

        return loss, metrics

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        probas, outputs = self(images)

        loss, metrics = self._compute_loss(probas, outputs, labels)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/gating_S", metrics.gating_entropy)

        # Log expert utilization (mean probability per expert)
        for name, prob in zip(self.expert_names, metrics.expert_probs_mean):
            self.log(f"train/P_{name}", prob)

        return loss

    def _hard_route(
        self, probas: torch.Tensor, outputs: list[torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard routing: pick the most probable expert per sample.

        Returns:
            preds: (batch_size,) predicted class indices
            acc:   scalar mean accuracy
        """
        all_outputs = torch.stack(outputs)  # (num_experts, batch_size, num_classes)
        expert_idx = probas.argmax(dim=1)  # (batch_size,)
        batch_size = labels.size(0)
        selected = all_outputs[expert_idx, torch.arange(batch_size)]
        preds = selected.argmax(dim=1)
        acc = (preds == labels).float().mean()
        return preds, acc

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        probas, outputs = self(images)

        loss, metrics = self._compute_loss(probas, outputs, labels)

        _, acc = self._hard_route(probas, outputs, labels)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/gating_S", metrics.gating_entropy, prog_bar=True)

        # Log expert probabilities (visible in progress bar)
        for name, prob in zip(self.expert_names, metrics.expert_probs_mean):
            self.log(f"val/P_{name}", prob, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        probas, outputs = self(images)

        _, acc = self._hard_route(probas, outputs, labels)

        self.log("test/acc", acc, prog_bar=True)

        # Cost gain: how much cheaper is the MoE vs the most expensive expert
        expected_cost = (probas * self.raw_costs).sum(dim=1).mean()
        cost_gain = 1 - expected_cost / self.raw_costs.max()
        self.log("test/cost_gain", cost_gain)

    def configure_optimizers(self):
        return build_optimizer(
            self.parameters(), self.hparams.optimizer, self.hparams.learning_rate
        )

    def get_expert_utilization(self, dataloader) -> dict[str, float]:
        """
        Compute expert utilization statistics on a dataset

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Dictionary mapping expert names to their utilization percentage
        """
        expert_counts = torch.zeros(len(self.experts))
        total_samples = 0

        with torch.no_grad(), eval_mode(self):
            for batch in dataloader:
                images, _ = batch
                images = images.to(self.device)
                probas, _ = self(images)

                # Count which expert has highest probability for each sample
                expert_choices = probas.argmax(dim=1)
                for idx in expert_choices:
                    expert_counts[idx] += 1

                total_samples += images.size(0)

        # Convert to pct
        utilization = {
            name: (count.item() / total_samples * 100)
            for name, count in zip(self.expert_names, expert_counts)
        }

        return utilization

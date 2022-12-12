import typing as T

import attr
import torch
import torch.nn.functional as F


class LossPreprocessing(torch.nn.Module):
    def __init__(self, inputs_are_logits: bool, apply_transform: bool):
        super(LossPreprocessing, self).__init__()

        self.inputs_are_logits = inputs_are_logits
        self.apply_transform = apply_transform

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        if self.inputs_are_logits:
            if (len(targets.unique()) > inputs.size(1)) or (targets.unique().max()+1 > inputs.size(1)):
                raise ValueError(
                    'The targets should be ordered values of equal length to the inputs 2nd dimension.'
                )
            if self.apply_transform:
                inputs = F.softmax(inputs, dim=1, dtype=inputs.dtype)
            targets = F.one_hot(
                targets.contiguous().view(-1), inputs.shape[1]
            ).float()
        else:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

        return inputs, targets


class TanimotoDistLoss(torch.nn.Module):
    """Tanimoto distance loss

    Reference:
        https://github.com/sentinel-hub/eo-flow/blob/master/eoflow/models/losses.py

    MIT License

    Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Matic Lubej, Grega Milčinski (Sinergise)
    Copyright (c) 2017-2020 Devis Peressutti, Jernej Puc, Anže Zupanc, Lojze Žust, Jovan Višnjić (Sinergise)
    """
    preprocessor = LossPreprocessing(
        inputs_are_logits=True,
        apply_transform=True
    )

    def __init__(
        self,
        volume: torch.Tensor,
        smooth: float = 1e-5,
        class_weights: T.Optional[torch.Tensor] = None
    ):
        super(TanimotoDistLoss, self).__init__()

        self.volume = volume
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        inputs, targets = self.preprocessor(inputs, targets)

        weights = torch.reciprocal(torch.square(self.volume))
        new_weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = torch.where(
            torch.isinf(weights), torch.ones_like(weights) * new_weights.max(), weights
        )
        intersection = (targets * inputs).sum(dim=0)
        sum_ = (targets * targets + inputs * inputs).sum(dim=0)
        num_ = (intersection * weights) + self.smooth
        den_ = ((sum_ - intersection) * weights) + self.smooth
        tanimoto = num_ / den_
        loss = 1.0 - tanimoto

        if self.class_weights is not None:
            loss = loss * self.class_weights

        return loss.sum()


class CrossEntropyLoss(torch.nn.Module):
    """Cross entropy loss
    """
    def __init__(
        self,
        reduction: T.Optional[str] = 'mean',
        class_weights: T.Optional[torch.Tensor] = None
    ):
        self.reduction= reduction
        self.class_weights = class_weights

        self.loss_func = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction=self.reduction
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(inputs, targets.contiguous().view(-1))


class FocalLoss(torch.nn.Module):
    """Focal loss

    Reference:
        https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    sigmoid = torch.nn.Sigmoid()
    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        reduction='none'
    )
    preprocessor = LossPreprocessing(
        inputs_are_logits=True,
        apply_transform=True
    )

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        inputs, targets = self.preprocessor(inputs, targets)
        ce_loss = self.cross_entropy_loss(
            inputs, targets.half()
        )
        ce_exp = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - ce_exp)**self.gamma * ce_loss

        return focal_loss.mean()


@attr.s
class QuantileLoss(object):
    """Loss function for quantile regression

    Reference:
        https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss

    THE MIT License

    Copyright 2020 Jan Beitner
    """
    quantiles: T.Tuple[float, float, float] = attr.ib(validator=attr.validators.instance_of(tuple))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Quantile loss (float)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - inputs[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.cat(losses, dim=1).sum(dim=1).mean()

        return loss


class WeightedL1Loss(torch.nn.Module):
    """Weighted L1Loss loss
    """
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        mae = torch.abs(
            inputs - targets
        )
        weight = inputs + targets
        loss = (mae * weight).mean()

        return loss


class MSELoss(torch.nn.Module):
    """MSE loss
    """
    loss_func = torch.nn.MSELoss()
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(
            inputs.contiguous().view(-1),
            targets.contiguous().view(-1)
        )

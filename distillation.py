import torch
from torch import nn
from torch.nn import functional as F


class KDLoss(nn.Module):
    """Knowledge Distillation loss."""

    def __init__(self, dim: int = -1, scale_T: bool = True) -> None:
        """Initializer for KDLoss.

        Args:
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
            scale_T (bool, optional): Whether to scale kldiv by T^2. Defaults to True.
        """
        super().__init__()

        self.dim = dim
        self.scale_T = scale_T

    def forward(self, pred: torch.Tensor, target: torch.Tensor, teacher_pred: torch.Tensor, T: float, alpha: float,
                beta: float = None) -> torch.Tensor:
        """Forward method for KDLoss.

        Args:
            pred (torch.Tensor): Predictions of student model. Tensor of shape (batch, num_classes).
            target (torch.Tensor): Labels. LongTensor of shape (batch,), containing class integers like [1, 2, 3, ...].
            teacher_pred (torch.Tensor): Predictions of teacher model. Tensor of shape (batch, num_classes).
            T (float): Temperature value for evaluating softmax.
            alpha (float): Weight for kldiv.
            beta (float, optional): Weight for crossentropy. If not provided (beta=None), will use beta = 1 - alpha. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """

        assert T >= 1.0, f"Expected temperature greater or equal to 1.0, but got {T}."

        if beta == None:
            assert alpha < 1.0, f"For weighted average (beta=None), alpha must be less than 1.0, but got {alpha}."
            beta = 1.0 - alpha

        # if self.scale_T:
        #    alpha = alpha

        pred_log_probs = F.log_softmax(pred / T, dim=self.dim)
        teacher_pred_log_probs = F.log_softmax(teacher_pred / T, dim=self.dim)

        kldiv = F.kl_div(pred_log_probs, teacher_pred_log_probs, log_target=True)
        crossentropy = F.cross_entropy(pred, target)

        return (alpha * kldiv + beta * crossentropy) * T * T

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNCLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, gamma=2.0, beta=2.0, eps=1e-6, warmup_steps=10):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.warmup_steps = warmup_steps

        self.register_buffer("FP", torch.zeros(num_classes))
        self.register_buffer("FN", torch.zeros(num_classes))
        self.register_buffer("step", torch.tensor(0))

    def forward(self, pred_logits, targets):
        device = pred_logits.device
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)
        self.step += 1

        probs = pred_logits.sigmoid()
        pt = targets * probs + (1 - targets) * (1 - probs)

        grad = torch.abs(probs.detach() - targets)
        pos_grad = grad * targets
        neg_grad = grad * (1 - targets)
        pos_grad_sum = pos_grad.sum(dim=0) + self.eps
        neg_grad_sum = neg_grad.sum(dim=0) + self.eps
        glr_weight = pos_grad_sum / (pos_grad_sum + neg_grad_sum)

        pred_binary = (probs > 0.5).float()
        fp = ((pred_binary == 1) & (targets == 0)).sum(dim=0).float()
        fn = ((pred_binary == 0) & (targets == 1)).sum(dim=0).float()

        self.FP = 0.9 * self.FP + 0.1 * fp.to(device)
        self.FN = 0.9 * self.FN + 0.1 * fn.to(device)

        if hasattr(self, "epoch") and self.epoch % 5 == 0:
            print(f"[GNC][Epoch {self.epoch}]")
            print("Top10 FP:", self.FP[:10].tolist())
            print("Top10 FN:", self.FN[:10].tolist())

        if self.step < self.warmup_steps:
            cbr_weight = torch.ones_like(glr_weight)
        else:
            bias_ratio = (self.FP + self.eps) / (self.FN + self.eps)
            x = torch.clamp(self.beta * (bias_ratio - 1.0), -10.0, 10.0)
            cbr_weight = (1 / (1 + torch.exp(-x))).clamp(0.01, 10.0)

        weight_glr = glr_weight.unsqueeze(0).expand_as(pred_logits)
        weight_cbr = cbr_weight.unsqueeze(0).expand_as(pred_logits).to(device)

        weight = self.alpha * weight_glr + (1 - self.alpha) * weight_cbr

        loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        loss = loss * weight

        if self.training and self.step <= 20:
            print(f"[GNC] step={self.step.item()} cls_loss={loss.mean().item():.2e}")
            print(f"      glr_w.mean={glr_weight.mean().item():.4f}, cbr_w.mean={cbr_weight.mean().item():.4f}, weight.mean={weight.mean().item():.4f}")

        return loss


# def get_classification_loss(name, epoch, warmup_epochs=30, num_classes=80, **kwargs):
#     if name == 'bce':
#         return nn.BCEWithLogitsLoss(reduction="none")
#     elif name == 'gnc':
#         return GNCLoss(num_classes=num_classes, **kwargs)
#     elif name == 'gnc_warmup_bce':
#         if epoch < warmup_epochs:
#             return nn.BCEWithLogitsLoss(reduction="none")
#         else:
#             return GNCLoss(num_classes=num_classes, **kwargs)
#     else:
#         raise ValueError(f"Unsupported loss: {name}")

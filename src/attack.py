# src/attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD:
    def __init__(self, model, eps=0.01, alpha=0.002, steps=10, random_start=True, device='cuda'):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def forward(self, x, y, toa):
        """
        x: [Batch, Frames, Objects, Features]
        y: Labels
        toa: Time of Accident
        """
        # --- 修改重点开始 ---
        # 原代码: self.model.eval()
        # 修改为: self.model.train()
        # 原因: cuDNN GRU 在 eval 模式下不支持计算梯度，而 PGD 需要计算梯度。
        self.model.train()
        # --- 修改重点结束 ---

        # Clone and detach to create a new tensor for perturbation
        x_adv = x.clone().detach().to(self.device)

        if self.random_start:
            # Add random noise within [-eps, eps]
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, min=x.min(), max=x.max())  # Clamp to valid range if needed

        for _ in range(self.steps):
            x_adv.requires_grad = True

            # Forward pass using the model
            losses, outputs, _ = self.model(x_adv, y, toa)

            # We use the total_loss (or specifically cross_entropy) to calculate gradients
            loss_cost = losses['cross_entropy'].mean()

            # Calculate gradients
            # retain_graph=False is fine here because we don't need to backprop through the model parameters,
            # we only need gradients w.r.t input x_adv.
            grad = torch.autograd.grad(loss_cost, x_adv, retain_graph=False, create_graph=False)[0]

            # Update adversarial images
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball (Projection step)
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=x.min(), max=x.max())

        # 注意：这里不需要再切回 eval()，因为 PGD 是在训练循环中调用的，
        # main.py 的后续代码会继续进行训练，需要模型保持 train 模式。
        self.model.train()
        return x_adv.detach()
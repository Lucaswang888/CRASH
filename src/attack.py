# src/attack.py (No Data Clamping Version)
import torch
import torch.nn as nn

class PGD:
    def __init__(self, model, teacher_model=None, eps=0.15, alpha=0.03, steps=10, 
                 random_start=True, device='cuda'):
        """
        初始化 PGD 攻击器 (无数据截断版)
        Args:
            model: Student 模型
            teacher_model: Teacher 模型
            eps: 攻击半径 (L_inf norm)
            alpha: 单步攻击步长
            steps: 迭代次数
            random_start: 是否随机初始化扰动
            device: 运行设备
        """
        self.model = model
        self.teacher_model = teacher_model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y, toa, nbatch=None):
        self.model.train() # 保持 Train 模式
        
        # 1. 获取 Teacher 输出 (Consistency Anchor)
        clean_target_out = None
        clean_target_feat = None
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                _, t_out, t_feat = self.teacher_model(x, y, toa, nbatch=nbatch)
                clean_target_out = torch.stack(t_out).detach()
                clean_target_feat = torch.stack(t_feat).detach()
        
        # 2. 获取 Student Clean 输出 (Stability Anchor)
        with torch.no_grad():
            _, s_out, s_feat = self.model(x, y, toa, nbatch=nbatch)
            clean_student_out = torch.stack(s_out).detach()
            clean_student_feat = torch.stack(s_feat).detach()

        # 3. 初始化扰动
        x_adv = x.clone().detach().to(self.device)
        
        if self.random_start:
            # 在 epsilon 球内随机初始化
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            # 这里的 x_adv 不需要截断，保留随机性

        # 4. 迭代攻击 (Four-Loss Maximization)
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward (Perturbed Student)
            _, adv_out, adv_feat = self.model(x_adv, y, toa, nbatch=nbatch)
            adv_stack_out = torch.stack(adv_out)
            adv_stack_feat = torch.stack(adv_feat)

            # --- 计算总攻击 Loss ---
            total_attack_loss = 0
            
            # (i) Output Stability
            total_attack_loss += self.mse_loss(adv_stack_out, clean_student_out)
            
            # (ii) Feature Stability
            total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_student_feat)

            # (iii & iv) Consistency
            if clean_target_out is not None:
                total_attack_loss += self.mse_loss(adv_stack_out, clean_target_out)
                total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_target_feat)

            # 梯度更新
            grad = torch.autograd.grad(total_attack_loss, x_adv, 
                                     retain_graph=False, create_graph=False)[0]
            
            # 梯度上升
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # --- 仅保留 Epsilon 投影 (核心约束) ---
            # 确保 |x_adv - x| <= eps
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x + delta
            
            # [已移除] 数据物理范围截断
            # x_adv = torch.clamp(x_adv, min=..., max=...) 

        self.model.train()
        return x_adv.detach()

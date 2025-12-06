# src/attack.py
import torch
import torch.nn as nn

class PGD:
    def __init__(self, model, teacher_model=None, eps=0.05, alpha=0.008, steps=10, random_start=True, device='cuda'):
        """
        参数建议 (基于你的数据分布 Mean=0.05, Std=0.17):
        eps: 0.05 (约 0.3倍 std, 覆盖均值)
        alpha: 0.008 (确保 10步能走完 eps)
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
        """
        x: [Batch, Frames, Objects, Features]
        y: Labels
        toa: Time of Accident
        nbatch: 用于模型内部状态重置的参数 (适配 main.py)
        """
        self.model.train() # 保持 train 模式以允许梯度回传
        
        # 1. 获取 Teacher 的参考输出 (作为 Consistency 的锚点)
        clean_target_out = None
        clean_target_feat = None
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                # 传入 nbatch 防止报错
                _, t_out, t_feat = self.teacher_model(x, y, toa, nbatch=nbatch)
                clean_target_out = torch.stack(t_out).detach()
                clean_target_feat = torch.stack(t_feat).detach()
        
        # 2. 获取 Student Clean 的参考输出 (作为 Stability 的锚点)
        # 这一步是为了让攻击者知道"原本的输出是什么"，从而最大化偏离
        with torch.no_grad():
            _, s_out, s_feat = self.model(x, y, toa, nbatch=nbatch)
            clean_student_out = torch.stack(s_out).detach()
            clean_student_feat = torch.stack(s_feat).detach()

        # 3. 初始化扰动
        x_adv = x.clone().detach().to(self.device)
        
        if self.random_start:
            # 随机初始化在 epsilon 球内
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            # 初始截断：保证物理意义 (Min=0)
            x_adv = torch.clamp(x_adv, min=0.0, max=6.0)

        # 4. 迭代攻击 (Four-Loss Attack)
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward (Perturbed Student)
            _, adv_out, adv_feat = self.model(x_adv, y, toa, nbatch=nbatch)
            adv_stack_out = torch.stack(adv_out)
            adv_stack_feat = torch.stack(adv_feat)

            # --- 计算四个 Loss 之和 ---
            # 目标：最大化 (Stability Loss + Consistency Loss)
            # 即寻找最能破坏"忠实度"的样本
            
            total_attack_loss = 0
            
            # (i) Output Stability: ||Student(Adv) - Student(Clean)||
            total_attack_loss += self.mse_loss(adv_stack_out, clean_student_out)
            
            # (ii) Feature Stability: ||Feat(Adv) - Feat(Clean)||
            # 权重 0.1 用于平衡量级 (Feature 维度通常较大)
            total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_student_feat)

            if clean_target_out is not None:
                # (iii) Output Consistency: ||Student(Adv) - Teacher||
                total_attack_loss += self.mse_loss(adv_stack_out, clean_target_out)
                
                # (iv) Feature Consistency: ||Feat(Adv) - Teacher_Feat||
                total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_target_feat)

            # 计算梯度
            grad = torch.autograd.grad(total_attack_loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # 更新扰动 (Gradient Ascent: sign() 实现 L_infinity 攻击)
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # 投影 (Projection)
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x + delta
            
            # 最终截断：必须保证数据物理合法性 (0.0 ~ 6.0)
            x_adv = torch.clamp(x_adv, min=0.0, max=6.0)

        # 恢复模型状态 (虽然 main.py 会再次调用 model.train()，这里保持良好习惯)
        self.model.train()
        
        return x_adv.detach()

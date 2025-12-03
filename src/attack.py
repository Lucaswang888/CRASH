import torch
import torch.nn as nn

class PGD:
    def __init__(self, model, teacher_model=None, eps=0.01, alpha=0.002, steps=10, random_start=True, device='cuda'):
        self.model = model
        self.teacher_model = teacher_model  # 需要传入 Teacher 模型
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y, toa):
        self.model.train()
        
        # 1. 获取 Teacher 的参考输出 (作为 Consistency 的锚点)
        # 注意：如果不想传 teacher_model，也可以把 teacher 的输出作为参数传进来
        clean_target_out = None
        clean_target_feat = None
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                _, t_out, t_feat = self.teacher_model(x, y, toa)
                clean_target_out = torch.stack(t_out).detach()
                clean_target_feat = torch.stack(t_feat).detach()
        
        # 2. 获取 Student Clean 的参考输出 (作为 Stability 的锚点)
        with torch.no_grad():
            _, s_out, s_feat = self.model(x, y, toa)
            clean_student_out = torch.stack(s_out).detach()
            clean_student_feat = torch.stack(s_feat).detach()

        # 3. 初始化扰动
        x_adv = x.clone().detach().to(self.device)
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, min=x.min(), max=x.max())

        # 4. 迭代攻击
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward (Perturbed Student)
            _, adv_out, adv_feat = self.model(x_adv, y, toa)
            adv_stack_out = torch.stack(adv_out)
            adv_stack_feat = torch.stack(adv_feat)

            # --- 计算四个 Loss ---
            total_attack_loss = 0
            
            # (i) Output Stability (S_pd): Student(Adv) vs Student(Clean)
            total_attack_loss += self.mse_loss(adv_stack_out, clean_student_out)
            
            # (ii) Feature Stability (S_ld): Student(Adv) vs Student(Clean)
            total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_student_feat)

            if clean_target_out is not None:
                # (iii) Output Consistency (C_ps): Student(Adv) vs Teacher
                total_attack_loss += self.mse_loss(adv_stack_out, clean_target_out)
                
                # (iv) Feature Consistency (C_lm): Student(Adv) vs Teacher
                total_attack_loss += 0.1 * self.mse_loss(adv_stack_feat, clean_target_feat)

            # 计算梯度 & 更新
            grad = torch.autograd.grad(total_attack_loss, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=x.min(), max=x.max())

        self.model.train()
        return x_adv.detach()

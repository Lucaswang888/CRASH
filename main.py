# main.py (Modified: Replace PGD Test with Parameter Perturbation)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os, time
import argparse
import shutil
import cv2
import copy  # 确保导入 copy 用于备份权重

from torch.utils.data import DataLoader
from src.Models import CRASH
from src.eval_tools import evaluation_P_R80, print_results, vis_results
from src.attack import PGD 
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def average_losses(losses_all):
    total_loss, cross_entropy, aux_loss = 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        aux_loss += losses['auxloss']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    return losses_mean

# Exponentially Weighted MSE
def weighted_mse_loss(input, target, toas, fps=20.0, weight_factor=5.0):
    T, B, C = input.shape
    device = input.device
    
    if toas.dim() > 1:
        toas = toas.squeeze(-1)

    weights = torch.ones((T, B), device=device)
    t_grid = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
    toas_exp = toas.unsqueeze(0).expand(T, B)
    
    dist_to_accident = (toas_exp - t_grid).float() / fps
    is_accident_video = (toas > 0).float().unsqueeze(0).expand(T, B)
    is_before_accident = (dist_to_accident >= 0).float()
    
    temporal_weight = torch.exp(-0.5 * torch.abs(dist_to_accident))
    
    final_weights = 1.0 + (weight_factor * temporal_weight * is_accident_video * is_before_accident)
    
    mse = (input - target) ** 2
    weighted_mse = mse * final_weights.unsqueeze(2)
    return weighted_mse.mean()

def test_all(testdata_loader, model):
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas,
                                                 hidden_in=None, nbatch=len(testdata_loader), testing=True)

            loss_val = p.loss_u1 / 2 * losses['cross_entropy']
            loss_val += p.loss_u2 / 2 * losses['auxloss']
            if 'log' in losses:
                loss_val += losses['log'].mean()
            losses['total_loss'] = loss_val
            losses_all.append(losses)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas, losses_all

    
def test_noise(testdata_loader, model, stddev=0.1, device=torch.device('cuda')):
    print(f">>> Running Input Gaussian Noise Test (std={stddev})...")
    all_pred = []
    all_labels = []
    all_toas = []

    model.eval()
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            noise = torch.randn_like(batch_xs).to(device) * stddev
            batch_xs_noisy = batch_xs + noise
            dummy_ys = torch.zeros_like(batch_ys).to(device)
            dummy_toas = torch.zeros_like(batch_toas).to(device)
            
            _, all_outputs, _ = model(batch_xs_noisy, dummy_ys, dummy_toas,
                                      hidden_in=None, nbatch=len(testdata_loader), 
                                      testing=True)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy() 
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas

# ------------------------------------------------------------------------------
# [NEW] Parameter Perturbation Test Function (Instead of PGD)
# ------------------------------------------------------------------------------
def test_param_perturbation(testdata_loader, model, device, stddev=0.02, target_layer_str='gru_net.gru'):
    """
    对指定层（如GRU）的参数注入高斯噪声，进行鲁棒性测试，并在测试后恢复权重。
    """
    print(f">>> Running Parameter Perturbation Test (Target: '{target_layer_str}', std={stddev})...")
    
    # 1. 备份原始权重 (Deep Copy)
    # 必须备份，否则模型权重会被永久破坏
    original_state_dict = copy.deepcopy(model.state_dict())
    
    # 2. 对参数注入噪声
    injected_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # 只有在 param 需要梯度且名字包含目标层字符串时才加噪
            if param.requires_grad and target_layer_str in name:
                noise = torch.randn_like(param).to(device) * stddev
                param.add_(noise) # 原地修改权重
                injected_count += 1
    
    if injected_count == 0:
        print(f"Warning: No layers found matching '{target_layer_str}'! No noise injected.")
    else:
        print(f"Injected noise into {injected_count} parameters.")

    # 3. 运行推理 (逻辑与 test_all 相同，但只取预测结果)
    all_pred = []
    all_labels = []
    all_toas = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(tqdm(testdata_loader, desc="Param Perturb Test")):
            # 注意：输入不加噪声，只有参数加了噪声
            _, all_outputs, _ = model(batch_xs, batch_ys, batch_toas, 
                                      hidden_in=None, nbatch=len(testdata_loader), testing=True)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    # 4. 恢复原始权重 (非常重要！)
    model.load_state_dict(original_state_dict)
    print(">>> Original weights restored.")
    
    return all_pred, all_labels, all_toas


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    aux_loss = losses['auxloss'].mean().item()

    log_dict = {
        'total_loss': total_loss,
        'cross_entropy': cross_entropy,
        'aux_loss': aux_loss,
        'lr': lr
    }
    if 'adv_loss' in losses:
        log_dict['adv_loss'] = losses['adv_loss'].item()
    if 'sim_loss' in losses:
        log_dict['sim_loss'] = losses['sim_loss'].item()
    if 'feat_loss' in losses:
        log_dict['feat_loss'] = losses['feat_loss'].item()

    logger.add_scalars("train/losses", log_dict, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics, prefix="test"):
    total_loss = losses['total_loss'].mean().item() if 'total_loss' in losses and not isinstance(losses['total_loss'], (int, float)) else 0
    cross_entropy = losses['cross_entropy'].mean() if 'cross_entropy' in losses and not isinstance(losses['cross_entropy'], (int, float)) else 0

    logger.add_scalars(f"{prefix}/losses/total_loss", {'total_loss': total_loss, 'cross_entropy': cross_entropy},
                       cur_iter)
    logger.add_scalars(f"{prefix}/accuracy/AP", {'AP': metrics['AP'], 'P_R80': metrics['P_R80']}, cur_iter)
    logger.add_scalars(f"{prefix}/accuracy/time-to-accident", {'mTTA': metrics['mTTA'], 'TTA_R80': metrics['TTA_R80']},
                       cur_iter)


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    start_epoch = 0
    if os.path.isfile(filename):
        print(f"==> Loading checkpoint from '{filename}'")
        checkpoint = torch.load(filename)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        
        state_dict = checkpoint['model']
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        if isTraining and optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f"Warning: No checkpoint found at '{filename}'")
    
    return model, optimizer, start_epoch

# ------------------------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------------------------

def train_eval():
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, p.dataset, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError 

    # Optimize DataLoader
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=False)

    # 1. Initialize Student Model (Base)
    model = CRASH(train_data.dim_feature, p.hidden_dim, p.latent_dim,
                  n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps,
                  with_saa=True)

    # 2. Initialize Teacher Model (If Robust Training)
    teacher_model = None
    if p.robust_train:
        print(">>> Initializing Teacher Model for Robustness Training...")
        teacher_model = CRASH(train_data.dim_feature, p.hidden_dim, p.latent_dim,
                              n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps,
                              with_saa=True)
        teacher_model = teacher_model.to(device)
        
        if p.pretrained_model and os.path.isfile(p.pretrained_model):
            load_checkpoint(teacher_model, optimizer=None, filename=p.pretrained_model, isTraining=False)
            print(f">>> Teacher Model loaded from: {p.pretrained_model}")
        else:
            raise ValueError("Error: For robust_train, you MUST provide a valid --pretrained_model (Clean Model) for the Teacher!")

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    # 3. Optimizer Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Load Student Weights [CRITICAL UPDATE HERE]
    start_epoch = -1
    if p.resume:
        model = model.to(device) 
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)
        print(f">>> Resuming Student from epoch {start_epoch}")
    
    elif p.robust_train:
        # ----------------------------------------------------------------------
        # [修改] 学生模型必须继承教师模型的权重，而不是从0开始
        # ----------------------------------------------------------------------
        print(f">>> [Robust Fine-tuning] Copying pretrained weights from Teacher to Student...")
        # 直接使用 load_state_dict 复制权重，保证完全一致
        model.load_state_dict(teacher_model.state_dict())
        model = model.to(device)
        # 注意：这里 optimizer 不需要加载，因为我们是要开始新的微调训练
    
    else:
        model = model.to(device)
        print(">>> Training Student from Scratch (Random Init)")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.train()

    iter_cur = 0
    best_metric = 0

    loss_rob = torch.nn.MSELoss() 

    for k in range(p.epoch):
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
            
        loop = tqdm(enumerate(traindata_loader), total=len(traindata_loader))

        for i, (batch_xs, batch_ys, batch_toas) in loop:
            optimizer.zero_grad()

            # --- 1. Standard Forward (Clean Data) ---
            losses, all_outputs, hidden_st_clean = model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

            total_loss = p.loss_u1 / 2 * losses['cross_entropy']
            total_loss += p.loss_u2 / 2 * losses['auxloss']
            if 'log' in losses:
                total_loss += losses['log'].mean()

            # --- 2. Robust Training Step ---
            if p.robust_train:
                current_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                
                # A. Generate PGD Attack
                pgd = PGD(current_model, eps=p.eps, alpha=p.alpha, steps=p.steps, device=device)
                
                perturbed_batch_xs = pgd.forward(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

                # B. Student Forward on Perturbed Data
                _, outputs_pgd, hidden_st_pgd = model(perturbed_batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

                # C. Teacher Forward (Consistency)
                with torch.no_grad():
                    _, outputs_th, hidden_st_teacher = teacher_model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

                # D. Calculate Losses
                stack_out_clean = torch.stack(all_outputs)
                stack_out_pgd = torch.stack(outputs_pgd)
                stack_out_teacher = torch.stack(outputs_th)

                stack_feat_clean = torch.stack(hidden_st_clean)
                stack_feat_pgd = torch.stack(hidden_st_pgd)
                stack_feat_teacher = torch.stack(hidden_st_teacher)

                # --- Loss Calculation ---
                # 使用 Softmax 后的概率计算 MSE
                prob_clean = torch.softmax(stack_out_clean, dim=-1)
                prob_teacher = torch.softmax(stack_out_teacher, dim=-1)
                prob_pgd = torch.softmax(stack_out_pgd, dim=-1)

                sim_loss = weighted_mse_loss(prob_clean, prob_teacher, batch_toas, 
                                             fps=train_data.fps, weight_factor=p.time_weight_factor) * p.sim_weight
                
                adv_loss = loss_rob(prob_pgd, prob_clean) * p.adv_weight

                # ... (前文计算 sim_loss, adv_loss 保持不变) ...

                # [修改前]
                # feat_consistency = loss_rob(stack_feat_clean, stack_feat_teacher)
                # feat_stability = loss_rob(stack_feat_pgd, stack_feat_clean)
                # feat_loss = (feat_consistency + feat_stability) * p.feat_weight

                # [修改后] 分别乘上各自的权重
                loss_Clm = loss_rob(stack_feat_clean, stack_feat_teacher) * p.feat_cons_weight
                loss_Sld = loss_rob(stack_feat_pgd, stack_feat_clean) * p.feat_stab_weight
                
                # 累加到总 Loss
                total_loss += adv_loss + sim_loss + loss_Clm + loss_Sld
                
                # 记录日志 (可选)
                losses['feat_cons'] = loss_Clm
                losses['feat_stab'] = loss_Sld
                losses['adv_loss'] = adv_loss
                losses['sim_loss'] = sim_loss

            losses['total_loss'] = total_loss

            losses['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loop.set_description(f"Epoch [{k}/{p.epoch}]")
            loop.set_postfix(loss=losses['total_loss'].item())

            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, iter_cur, losses, lr)

            iter_cur += 1
            # ... Inside train_eval loop ...
            if iter_cur % p.test_iter == 0:
                model.eval()
                # 1. Clean Test
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
                loss_val = average_losses(losses_all)
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
                write_test_scalars(logger, k, iter_cur, loss_val, metrics, prefix="test_clean")
                
                print(f"\n[Epoch {k} Iter {iter_cur}] Clean AP: {metrics['AP']:.4f} | mTTA: {metrics['mTTA']:.4f}")

                # 2. Noise Test
                if p.robust_train:
                    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std, device=device)
                    metrics_n = {}
                    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
                    write_test_scalars(logger, k, iter_cur, {}, metrics_n, prefix="test_noise")
                    
                    print(f"[Epoch {k} Iter {iter_cur}] Noisy AP: {metrics_n['AP']:.4f} | mTTA: {metrics_n['mTTA']:.4f}")

                model.train()

        model_file = os.path.join(model_dir, 'model_%02d.pth' % (k))
        state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        
        torch.save({'epoch': k,
                    'model': state_dict_to_save,
                    'optimizer': optimizer.state_dict()}, model_file)

        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))
            if p.robust_train:
                update_final_model(model_file, os.path.join(model_dir, 'final_model_rob.pth'))

        scheduler.step(losses['total_loss'])
    logger.close()


def update_final_model(src_file, dest_file):
    assert os.path.exists(src_file), "src file does not exist!"
    if os.path.exists(dest_file):
        os.remove(dest_file)
    shutil.copyfile(src_file, dest_file)


def test_eval():
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    result_dir = os.path.join(p.output_dir, p.dataset, 'test_results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError

    # Optimize DataLoader
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=False)

    model = CRASH(test_data.dim_feature, p.hidden_dim, p.latent_dim,
                        n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps,
                        with_saa=True)
    model = model.to(device)

    if os.path.isfile(p.model_file):
        print(f"Loading checkpoint: {p.model_file}")
        checkpoint = torch.load(p.model_file)
        state_dict = checkpoint['model']
        
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"Error: No checkpoint found at {p.model_file}")
        return

    model.eval()

    print("------------------------------------------------")
    print(">>> Running Standard Evaluation (Clean Data)...")
    all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
    
    all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
    video_ap = average_precision_score(all_labels, all_vid_scores)

    metrics = {}
    metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
    
    print(f"[Clean Results]\nVideo AP: {video_ap:.4f} | Frame AP: {metrics['AP']:.4f} | mTTA: {metrics['mTTA']:.4f} | TTA_R80: {metrics['TTA_R80']:.4f} | P_R80: {metrics['P_R80']:.4f}")
    
    print("------------------------------------------------")
    print(f">>> Running Robustness Evaluation (Gaussian Noise std={p.noise_std})...")
    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std, device=device)
    
    all_vid_scores_n = [max(pred[:int(toa)]) for toa, pred in zip(all_toas_n, all_pred_n)]
    video_ap_n = average_precision_score(all_labels_n, all_vid_scores_n)

    metrics_n = {}
    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
    
    print(f"[Noisy Results]\nVideo AP: {video_ap_n:.4f} | Frame AP: {metrics_n['AP']:.4f} | mTTA: {metrics_n['mTTA']:.4f} | TTA_R80: {metrics_n['TTA_R80']:.4f} | P_R80: {metrics_n['P_R80']:.4f}")
    
    print("------------------------------------------------")
    
    # --------------------------------------------------------
    # [MODIFIED] Replaced PGD Attack with Parameter Perturbation
    # --------------------------------------------------------
    # 针对 GRU 层的参数干扰 (模拟 FTS 的 Intermediate Layer Perturbation)
    # stddev 可以设为 0.02 (论文常用值) 或者使用 p.eps
    param_noise_std = 0.2
    target_layer_name = 'gru_net.gru'
    
    # 调用新函数
    all_pred_ilp, all_labels_ilp, all_toas_ilp = test_param_perturbation(
        testdata_loader, model, device, stddev=param_noise_std, target_layer_str=target_layer_name
    )
    
    all_vid_scores_ilp = [max(pred[:int(toa)]) for toa, pred in zip(all_toas_ilp, all_pred_ilp)]
    video_ap_ilp = average_precision_score(all_labels_ilp, all_vid_scores_ilp)
    
    metrics_ilp = {}
    metrics_ilp['AP'], metrics_ilp['mTTA'], metrics_ilp['TTA_R80'], metrics_ilp['P_R80'] = evaluation_P_R80(all_pred_ilp, all_labels_ilp, all_toas_ilp, fps=test_data.fps)
    
    print(f"[Param Perturbation ({target_layer_name}) Results]\nVideo AP: {video_ap_ilp:.4f} | Frame AP: {metrics_ilp['AP']:.4f} | mTTA: {metrics_ilp['mTTA']:.4f}")
    
    print("------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_dir', type=str, default='./rub_output_new', help='The directory to save the output results.')
    parser.add_argument('--data_path', type=str, default='./data', help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='crash', choices=['dad', 'crash', 'a3d'],
                        help='The name of dataset.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='The base learning rate.')
    parser.add_argument('--epoch', type=int, default=80, help='The number of training epoches.')
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size. Recommended: 32')
    parser.add_argument('--num_rnn', type=int, default=2, help='RNN cells.')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='Feature embedding methods.')
    parser.add_argument('--test_iter', type=int, default=64, help='Iteration to perform evaluation.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dim.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dim.')
    parser.add_argument('--loss_u1', type=float, default=1, help='Weighting factor aux loss.')
    parser.add_argument('--loss_u2', type=float, default=15, help='Weighting factor aux loss.')
    parser.add_argument('--gpus', type=str, default="1", help="GPU IDs.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'], help='Phase.')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all epochs.')
    parser.add_argument('--visualize', action='store_true', help='Visualization flag.')
    parser.add_argument('--resume', action='store_true', help='Resume training.')
    parser.add_argument('--model_file', type=str, default='./output/CRASH/vgg16/dad/snapshot/model_23.pth',
                        help='Model file to save to (Student) or Resume from.')

    # --- ROBUSTNESS ARGS (SOTA Config) ---
    parser.add_argument('--robust_train', action='store_true', help='Enable Adversarial Training (PGD).')
    parser.add_argument('--pretrained_model', type=str, default=None, 
                        help='Path to the Clean Pretrained Model (Teacher) weights. REQUIRED for robust_train.')
    
    parser.add_argument('--eps', type=float, default=0.02, help='PGD epsilon (perturbation magnitude).')
    parser.add_argument('--alpha', type=float, default=0.003, help='PGD alpha (step size).')
    parser.add_argument('--steps', type=int, default=10, help='PGD number of steps.')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight for adversarial loss (Output Level).')
    parser.add_argument('--sim_weight', type=float, default=0.5, help='Weight for similarity loss (Output Level).')
    parser.add_argument('--feat_weight', type=float, default=0.05, help='Weight for feature consistency/stability loss (Hidden Level).')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Stddev for Gaussian noise testing.')
    parser.add_argument('--feat_cons_weight', type=float, default=0.05, 
                        help='Weight for Latent Manifold Consistency (Clm)')
    parser.add_argument('--feat_stab_weight', type=float, default=0.05, 
                        help='Weight for Latent Dynamics Stability (Sld)')
    
    # [New Arg] Time Weight Factor
    parser.add_argument('--time_weight_factor', type=float, default=5.0, 
                        help='Weight factor for temporal consistency near accident.')

    p = parser.parse_args()

    if p.phase == 'test':
        test_eval()

    else:
        train_eval()

# main.py (Updated)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil
import cv2
import copy  # For Teacher Model

from torch.utils.data import DataLoader
from src.Models import CRASH
from src.eval_tools import evaluation_P_R80, print_results, vis_results
from src.attack import PGD  # IMPORT THE NEW PGD CLASS
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)


# ... (Include average_losses, preprocess_results, draw_curve, get_video_frames, draw_anchors, test_all_vis functions here - unchanged) ...
# 为了节省篇幅，省略了未修改的辅助函数，请保留原文件中的这些函数。
# 重点修改 average_losses 之后的 test_all 和 新增的 test_noise

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


def test_all(testdata_loader, model):
    # Unchanged logic, just ensure it handles the model correctly
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas,
                                                 hidden_in=None, nbatch=len(testdata_loader), testing=False)

            # Re-calculate total loss for logging
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


# --- NEW FUNCTION: Test with Noise (Based on Reference Code) ---
def test_noise(testdata_loader, model, stddev=0.1, device=torch.device('cuda')):
    print(f"Running Test with Gaussian Noise (std={stddev})...")
    all_pred = []
    all_labels = []
    all_toas = []

    model.eval()
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            # Add Gaussian Noise
            noise = torch.normal(mean=0.0, std=stddev, size=batch_xs.size()).to(device)
            batch_xs_noisy = batch_xs + noise

            _, all_outputs, _ = model(batch_xs_noisy, batch_ys, batch_toas,
                                      hidden_in=None, nbatch=len(testdata_loader), testing=False)

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


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    aux_loss = losses['auxloss'].mean().item()

    # Check for robust losses
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

    logger.add_scalars("train/losses", log_dict, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics, prefix="test"):
    total_loss = losses['total_loss'].mean().item() if 'total_loss' in losses else 0
    cross_entropy = losses['cross_entropy'].mean() if 'cross_entropy' in losses else 0

    logger.add_scalars(f"{prefix}/losses/total_loss", {'total_loss': total_loss, 'cross_entropy': cross_entropy},
                       cur_iter)
    logger.add_scalars(f"{prefix}/accuracy/AP", {'AP': metrics['AP'], 'P_R80': metrics['P_R80']}, cur_iter)
    logger.add_scalars(f"{prefix}/accuracy/time-to-accident", {'mTTA': metrics['mTTA'], 'TTA_R80': metrics['TTA_R80']},
                       cur_iter)


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, start_epoch


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

    # Dataset Setup
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError  # Simplified for brevity

    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    model = CRASH(train_data.dim_feature, p.hidden_dim, p.latent_dim,
                  n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps,
                  with_saa=True)

    # --- Robustness: Teacher Model Initialization ---
    if p.robust_train:
        print("Initializing Teacher Model for Robustness Training...")
        teacher_model = copy.deepcopy(model)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train()

    start_epoch = -1
    if p.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)

    iter_cur = 0
    best_metric = 0

    # Loss functions for consistency (from reference)
    loss_rob = torch.nn.L1Loss()  # Or MSELoss depending on feature scale

    for k in range(p.epoch):
        loop = tqdm(enumerate(traindata_loader), total=len(traindata_loader))
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue

        for i, (batch_xs, batch_ys, batch_toas) in loop:
            optimizer.zero_grad()

            # --- Normal Training Step ---
            # Forward on Clean Data
            losses, all_outputs, hidden_st = model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

            # Base Loss Calculation
            total_loss = p.loss_u1 / 2 * losses['cross_entropy']
            total_loss += p.loss_u2 / 2 * losses['auxloss']
            if 'log' in losses:
                total_loss += losses['log'].mean()

            # --- Robust Training Step (Adversarial) ---
            if p.robust_train:
                # 1. Generate PGD Attack
                # Note: PGD needs to know which model to attack.
                # If DataParallel, use model.module.
                current_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                pgd = PGD(current_model, eps=p.eps, alpha=p.alpha, steps=p.steps, device=device)

                # Generate perturbed input
                perturbed_batch_xs = pgd.forward(batch_xs, batch_ys, batch_toas)

                # 2. Forward on Perturbed Data (Student/Current Model)
                losses_pgd, outputs_pgd, _ = model(perturbed_batch_xs, batch_ys, batch_toas,
                                                   nbatch=len(traindata_loader))

                # 3. Forward on Clean Data (Teacher Model)
                # We want the student's perturbed output to match the teacher's clean output
                with torch.no_grad():
                    losses_th, outputs_th, _ = teacher_model(batch_xs, batch_ys, batch_toas,
                                                             nbatch=len(traindata_loader))

                # 4. Calculate Robust Losses (Inspired by reference code)
                # Loss 1: Classification on perturbed data
                # loss_cls_pgd = losses_pgd['cross_entropy'].mean()

                # Loss 2: Adversarial Loss (Distance between PGD output and Clean Output)
                # Reference: adv_loss = loss_rob(outputs_PGD, outputs_nos) * args.adv_loss
                # We need to flatten outputs for L1Loss. CRASH outputs are lists of tensors [frames]
                # Let's take the last frame or average
                # CRASH output is a list of [Batch, 2]. Let's stack them.
                stack_out_pgd = torch.stack(outputs_pgd)  # [Frames, Batch, 2]
                stack_out_clean = torch.stack(all_outputs)  # [Frames, Batch, 2]
                stack_out_teacher = torch.stack(outputs_th)  # [Frames, Batch, 2]

                # Adversarial Loss: Output(Perturbed) vs Output(Clean Student)
                adv_loss = loss_rob(stack_out_pgd, stack_out_clean) * p.adv_weight

                # Consistency Loss: Output(Clean Student) vs Output(Clean Teacher)
                sim_loss = loss_rob(stack_out_clean, stack_out_teacher) * p.sim_weight

                # Add to total loss
                total_loss += adv_loss + sim_loss

                # Log these for Tensorboard
                losses['adv_loss'] = adv_loss
                losses['sim_loss'] = sim_loss

            losses['total_loss'] = total_loss

            # Backward
            losses['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loop.set_description(f"Epoch [{k}/{p.epoch}]")
            loop.set_postfix(loss=losses['total_loss'].item())

            # Update Teacher Model (EMA - Exponential Moving Average is better, but reference uses fixed or simple copy)
            # If following strict "Distillation" logic from reference, teacher might be fixed pre-trained.
            # Or we can update teacher occasionally. Here we skip complex EMA for simplicity unless required.

            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, iter_cur, losses, lr)

            iter_cur += 1
            if iter_cur % p.test_iter == 0:
                model.eval()
                # 1. Standard Test
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
                loss_val = average_losses(losses_all)
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred,
                                                                                                        all_labels,
                                                                                                        all_toas,
                                                                                                        fps=test_data.fps)
                write_test_scalars(logger, k, iter_cur, loss_val, metrics, prefix="test_clean")

                # 2. Noise Robustness Test (Optional, every few epochs or same iter)
                if p.robust_train:
                    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std,
                                                                      device=device)
                    metrics_n = {}
                    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(
                        all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
                    write_test_scalars(logger, k, iter_cur, {}, metrics_n, prefix="test_noise")

                model.train()

        model_file = os.path.join(model_dir, 'model_%02d.pth' % (k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(p.gpus) > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)

        # Save best based on Clean AP
        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))
            # Also save best model as robust checkpoint if robust training
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

    # 1. Load Data
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError

    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    # 2. Build Model
    model = CRASH(test_data.dim_feature, p.hidden_dim, p.latent_dim,
                        n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps,
                        with_saa=True)
    model = model.to(device)

    # 3. Load Checkpoint (保持您原本的逻辑，不含特殊修复)
    if os.path.isfile(p.model_file):
        print(f"Loading checkpoint: {p.model_file}")
        checkpoint = torch.load(p.model_file)
        state_dict = checkpoint['model']
        
        # 处理 DataParallel 的 module. 前缀 (标准操作)
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

    # 4. Standard Evaluation (Clean)
    print("------------------------------------------------")
    print(">>> Running Standard Evaluation (Clean Data)...")
    all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
    
    # --- [关键补充] 计算 Clean Video AP ---
    # 取每个视频在事故发生前(TOA)的最大预测值作为该视频的得分
    all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
    video_ap = average_precision_score(all_labels, all_vid_scores)
    # ------------------------------------

    metrics = {}
    metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
    
    # 打印结果 (增加了 Video AP)
    print(f"[Clean Results]\nVideo AP: {video_ap:.4f} | Frame AP: {metrics['AP']:.4f} | mTTA: {metrics['mTTA']:.4f} | TTA_R80: {metrics['TTA_R80']:.4f} | P_R80: {metrics['P_R80']:.4f}")
    
    # 5. Robustness Evaluation (Noise)
    print("------------------------------------------------")
    print(f">>> Running Robustness Evaluation (Gaussian Noise std={p.noise_std})...")
    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std, device=device)
    
    # --- [关键补充] 计算 Noisy Video AP ---
    all_vid_scores_n = [max(pred[:int(toa)]) for toa, pred in zip(all_toas_n, all_pred_n)]
    video_ap_n = average_precision_score(all_labels_n, all_vid_scores_n)
    # ------------------------------------

    metrics_n = {}
    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
    
    # 打印结果 (增加了 Video AP)
    print(f"[Noisy Results]\nVideo AP: {video_ap_n:.4f} | Frame AP: {metrics_n['AP']:.4f} | mTTA: {metrics_n['mTTA']:.4f} | TTA_R80: {metrics_n['TTA_R80']:.4f} | P_R80: {metrics_n['P_R80']:.4f}")
    print("------------------------------------------------")

# --- Updated Argument Parser ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (Original Args) ...
    parser.add_argument('--output_dir', type=str, default='./rub_output', help='The directory to save the output results.')
    parser.add_argument('--data_path', type=str, default='./data', help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='crash', choices=['dad', 'crash', 'a3d'],
                        help='The name of dataset.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='The base learning rate.')
    parser.add_argument('--epoch', type=int, default=80, help='The number of training epoches.')
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size.')
    parser.add_argument('--num_rnn', type=int, default=2, help='RNN cells.')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='Feature embedding methods.')
    parser.add_argument('--test_iter', type=int, default=64, help='Iteration to perform evaluation.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dim.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dim.')
    parser.add_argument('--loss_u1', type=float, default=1, help='Weighting factor aux loss.')
    parser.add_argument('--loss_u2', type=float, default=15, help='Weighting factor aux loss.')
    parser.add_argument('--gpus', type=str, default="0", help="GPU IDs.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'], help='Phase.')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all epochs.')
    parser.add_argument('--visualize', action='store_true', help='Visualization flag.')
    parser.add_argument('--resume', action='store_true', help='Resume training.')
    parser.add_argument('--model_file', type=str, default='./output/CRASH/vgg16/dad/snapshot/model_23.pth',
                        help='Model file.')

    # --- NEW ROBUSTNESS ARGS ---
    parser.add_argument('--robust_train', action='store_true', help='Enable Adversarial Training (PGD).')
    parser.add_argument('--eps', type=float, default=0.01, help='PGD epsilon (perturbation magnitude).')
    parser.add_argument('--alpha', type=float, default=0.002, help='PGD alpha (step size).')
    parser.add_argument('--steps', type=int, default=5, help='PGD number of steps.')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight for adversarial loss.')
    parser.add_argument('--sim_weight', type=float, default=0.5, help='Weight for similarity/consistency loss.')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Stddev for Gaussian noise testing.')

    p = parser.parse_args()

    if p.phase == 'test':
        test_eval()

    else:
        train_eval()
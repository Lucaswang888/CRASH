# main.py (Final Corrected Version with Feature-Level Stability)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil
import cv2
import copy  

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


def test_all(testdata_loader, model):
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas,
                                                 hidden_in=None, nbatch=len(testdata_loader), testing=True)

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

    
def test_noise(testdata_loader, model, stddev=0.1, device=torch.device('cuda')):
    print(f"Running Test with Gaussian Noise (std={stddev})...")
    all_pred = []
    all_labels = []
    all_toas = []

    model.eval()
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            # 1. 生成噪声 (Blind Test)
            print(">>> [WARNING] 执行全盲测试：输入全为随机噪声，不含任何原图信息！")
            batch_xs_noisy = torch.randn_like(batch_xs).to(device) * stddev
            
            # 2. 制造假标签 (Dummy Labels/TOAs) 防止Leakage
            dummy_ys = torch.zeros_like(batch_ys).to(device)
            dummy_toas = torch.zeros_like(batch_toas).to(device)
            
            # 3. Forward
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
            
            # Save results for AP calc using REAL labels
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

    log_dict = {
        'total_loss': total_loss,
        'cross_entropy': cross_entropy,
        'aux_loss': aux_loss,
        'lr': lr
    }
    # Log robust losses
    if 'adv_loss' in losses:
        log_dict['adv_loss'] = losses['adv_loss'].item()
    if 'sim_loss' in losses:
        log_dict['sim_loss'] = losses['sim_loss'].item()
    # Log feature losses
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
        raise NotImplementedError 

    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

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

        # Freeze Teacher
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    # 3. Optimizer Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Load Student Weights
    start_epoch = -1
    if p.resume:
        model = model.to(device) 
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)
        print(f">>> Resuming Student from epoch {start_epoch}")
    elif p.robust_train and p.pretrained_model:
        model = model.to(device)
        load_checkpoint(model, optimizer=None, filename=p.pretrained_model, isTraining=False)
        print(f">>> Initialized Student with Pretrained Clean Weights from {p.pretrained_model}")
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
            # Capture hidden_st_clean (Feature Level)
            losses, all_outputs, hidden_st_clean = model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

            # Base Loss Calculation
            total_loss = p.loss_u1 / 2 * losses['cross_entropy']
            total_loss += p.loss_u2 / 2 * losses['auxloss']
            if 'log' in losses:
                total_loss += losses['log'].mean()

            # --- 2. Robust Training Step (Adversarial + Feature Consistency) ---
            if p.robust_train:
                current_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                
                # A. Generate PGD Attack
                pgd = PGD(current_model, eps=p.eps, alpha=p.alpha, steps=p.steps, device=device)
                perturbed_batch_xs = pgd.forward(batch_xs, batch_ys, batch_toas)

                # B. Student Forward on Perturbed Data (Capture hidden_st_pgd)
                _, outputs_pgd, hidden_st_pgd = model(perturbed_batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

                # C. Teacher Forward on Clean Data (Capture hidden_st_teacher)
                with torch.no_grad():
                    _, outputs_th, hidden_st_teacher = teacher_model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

                # D. Calculate Losses
                # Output Level
                stack_out_clean = torch.stack(all_outputs)
                stack_out_pgd = torch.stack(outputs_pgd)
                stack_out_teacher = torch.stack(outputs_th)

                # Feature Level (Hidden States)
                # CRASH returns hidden states as a list [Batch, Hidden_Dim]. Need to stack to [Frames, Batch, Hidden_Dim]
                stack_feat_clean = torch.stack(hidden_st_clean)
                stack_feat_pgd = torch.stack(hidden_st_pgd)
                stack_feat_teacher = torch.stack(hidden_st_teacher)

                # --- Loss Components ---
                
                # 1. Output Adversarial Loss (Student_PGD vs Teacher_Clean)
                adv_loss = loss_rob(stack_out_pgd, stack_out_teacher) * p.adv_weight

                # 2. Output Similarity Loss (Student_Clean vs Teacher_Clean)
                sim_loss = loss_rob(stack_out_clean, stack_out_teacher) * p.sim_weight

                # 3. Feature Level Stability & Consistency
                # Enforce: Student Features (Clean & Perturbed) should match Teacher Features (Clean)
                feat_consistency = loss_rob(stack_feat_clean, stack_feat_teacher)
                feat_stability = loss_rob(stack_feat_pgd, stack_feat_teacher)
                feat_loss = (feat_consistency + feat_stability) * p.feat_weight

                total_loss += adv_loss + sim_loss + feat_loss
                
                losses['adv_loss'] = adv_loss
                losses['sim_loss'] = sim_loss
                losses['feat_loss'] = feat_loss

            losses['total_loss'] = total_loss

            losses['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loop.set_description(f"Epoch [{k}/{p.epoch}]")
            loop.set_postfix(loss=losses['total_loss'].item())

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

                # 2. Noise Robustness Test
                if p.robust_train:
                    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std,
                                                                      device=device)
                    metrics_n = {}
                    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(
                        all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
                    write_test_scalars(logger, k, iter_cur, {}, metrics_n, prefix="test_noise")

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

    # 3. Load Checkpoint
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

    # 4. Standard Evaluation (Clean)
    print("------------------------------------------------")
    print(">>> Running Standard Evaluation (Clean Data)...")
    all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
    
    # Calculate Clean Video AP
    all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
    video_ap = average_precision_score(all_labels, all_vid_scores)

    metrics = {}
    metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
    
    print(f"[Clean Results]\nVideo AP: {video_ap:.4f} | Frame AP: {metrics['AP']:.4f} | mTTA: {metrics['mTTA']:.4f} | TTA_R80: {metrics['TTA_R80']:.4f} | P_R80: {metrics['P_R80']:.4f}")
    
    # 5. Robustness Evaluation (Noise)
    print("------------------------------------------------")
    print(f">>> Running Robustness Evaluation (Gaussian Noise std={p.noise_std})...")
    all_pred_n, all_labels_n, all_toas_n = test_noise(testdata_loader, model, stddev=p.noise_std, device=device)
    
    # Calculate Noisy Video AP
    all_vid_scores_n = [max(pred[:int(toa)]) for toa, pred in zip(all_toas_n, all_pred_n)]
    video_ap_n = average_precision_score(all_labels_n, all_vid_scores_n)

    metrics_n = {}
    metrics_n['AP'], metrics_n['mTTA'], metrics_n['TTA_R80'], metrics_n['P_R80'] = evaluation_P_R80(all_pred_n, all_labels_n, all_toas_n, fps=test_data.fps)
    
    print(f"[Noisy Results]\nVideo AP: {video_ap_n:.4f} | Frame AP: {metrics_n['AP']:.4f} | mTTA: {metrics_n['mTTA']:.4f} | TTA_R80: {metrics_n['TTA_R80']:.4f} | P_R80: {metrics_n['P_R80']:.4f}")
    print("------------------------------------------------")

# --- Updated Argument Parser ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_dir', type=str, default='./rub_output_new', help='The directory to save the output results.')
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
    parser.add_argument('--gpus', type=str, default="1", help="GPU IDs.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'], help='Phase.')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all epochs.')
    parser.add_argument('--visualize', action='store_true', help='Visualization flag.')
    parser.add_argument('--resume', action='store_true', help='Resume training.')
    parser.add_argument('--model_file', type=str, default='./output/CRASH/vgg16/dad/snapshot/model_23.pth',
                        help='Model file to save to (Student) or Resume from.')

    # --- ROBUSTNESS ARGS ---
    parser.add_argument('--robust_train', action='store_true', help='Enable Adversarial Training (PGD).')
    parser.add_argument('--pretrained_model', type=str, default=None, 
                        help='Path to the Clean Pretrained Model (Teacher) weights. REQUIRED for robust_train.')
    
    parser.add_argument('--eps', type=float, default=0.01, help='PGD epsilon (perturbation magnitude).')
    parser.add_argument('--alpha', type=float, default=0.002, help='PGD alpha (step size).')
    parser.add_argument('--steps', type=int, default=5, help='PGD number of steps.')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight for adversarial loss (Output Level).')
    parser.add_argument('--sim_weight', type=float, default=0.5, help='Weight for similarity loss (Output Level).')
    parser.add_argument('--feat_weight', type=float, default=0.1, help='Weight for feature consistency/stability loss (Hidden Level).')
    parser.add_argument('--noise_std', type=float, default=2, help='Stddev for Gaussian noise testing.')

    p = parser.parse_args()

    if p.phase == 'test':
        test_eval()

    else:
        train_eval()

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

def evaluation_train(all_pred, all_labels, time_of_accidents, fps=20.0):
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)] 
        else:
            pred = all_pred[idx, :]
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0 
        for i in range(len(preds_eval)):
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    new_index = np.argsort(Recall)
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
    new_Time[-1] = Time[rep_index[-1]]

    mTTA = np.mean(new_Time) * total_seconds


    return mTTA


def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]
        else:
            pred = all_pred[idx, :]
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0 
        for i in range(len(preds_eval)):
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0: 
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]

    # 去掉重复的 Recall 点
    _, rep_index = np.unique(Recall, return_index=True)

    # === 防止 rep_index 为空或只剩 1 个元素 ===
    if rep_index.size <= 1:
        # 曲线退化：所有样本的 Recall 基本一样
        AP = 0.0
        mTTA = 0.0
        P_R80 = 0.0
        TTA_R80 = 0.0
        return AP, mTTA, TTA_R80, P_R80

    # 原来作者丢弃第一个 index
    rep_index = rep_index[1:]

    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    # ---- 计算 AP ----
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i-1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i-1]) / 2.0

    mTTA = np.mean(new_Time) * total_seconds

    # ---- 计算 P_R80 / TTA_R80，也要防止没有 Recall≥0.8 的情况 ----
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)

    idx_80 = np.where(new_Recall >= 0.8)[0]
    if idx_80.size == 0:
        # 如果曲线上从来没到过 0.8，就取最后一个点
        P_R80 = new_Precision[-1]
        TTA_R80 = sort_time[-1] * total_seconds
    else:
        P_R80 = new_Precision[idx_80[0]]
        TTA_R80 = sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds

    return AP, mTTA, TTA_R80, P_R80


def evaluation_P_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
    # ===== 添加输入检查 =====
    print(f"\n{'=' * 50}")
    print(f"Evaluation Debug Information")
    print(f"{'=' * 50}")
    print(f"Input shapes:")
    print(f"  - all_pred: {all_pred.shape}")
    print(f"  - all_labels: {all_labels.shape}")
    print(f"  - time_of_accidents: {time_of_accidents.shape}")
    print(f"\nLabel distribution:")
    print(f"  - Positive samples: {np.sum(all_labels)} / {len(all_labels)}")
    print(f"  - Negative samples: {len(all_labels) - np.sum(all_labels)} / {len(all_labels)}")
    print(f"\nPrediction statistics:")
    print(f"  - Min: {np.min(all_pred):.6f}")
    print(f"  - Max: {np.max(all_pred):.6f}")
    print(f"  - Mean: {np.mean(all_pred):.6f}")
    print(f"  - Std: {np.std(all_pred):.6f}")
    print(f"  - Median: {np.median(all_pred):.6f}")

    preds_eval = []
    min_pred = np.inf
    max_pred = -np.inf
    n_frames = 0

    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]
        else:
            pred = all_pred[idx, :]
        min_pred = min(min_pred, np.min(pred))
        max_pred = max(max_pred, np.max(pred))
        preds_eval.append(pred)
        n_frames += len(pred)

    print(f"\nPrediction range after TOA filtering:")
    print(f"  - Min: {min_pred:.6f}")
    print(f"  - Max: {max_pred:.6f}")
    print(f"  - Total frames: {n_frames}")

    total_seconds = all_pred.shape[1] / fps

    # 检查阈值范围
    threshold_start = max(min_pred, 0)
    threshold_end = 1.0
    threshold_step = 0.001
    threshold_range = np.arange(threshold_start, threshold_end, threshold_step)

    print(f"\nThreshold scanning:")
    print(f"  - Start: {threshold_start:.6f}")
    print(f"  - End: {threshold_end:.6f}")
    print(f"  - Step: {threshold_step}")
    print(f"  - Total steps: {len(threshold_range)}")

    if len(threshold_range) == 0:
        print("\n⚠️  WARNING: No threshold range to scan!")
        print(f"   This happens when min_pred ({min_pred:.6f}) >= 1.0")
        return 0.0, 0.0, 0.0, 0.0

    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0

    # 记录前几个阈值的详细信息
    debug_thresholds = []

    for Th in threshold_range:
        Tp = 0.0
        Tp_Fp = 0.0
        time = 0.0
        counter = 0.0

        for i in range(len(preds_eval)):
            tp = np.where(preds_eval[i] * all_labels[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)

        if Tp_Fp == 0:
            continue
        else:
            Precision[cnt] = Tp / Tp_Fp

        if np.sum(all_labels) == 0:
            continue
        else:
            Recall[cnt] = Tp / np.sum(all_labels)

        if counter == 0:
            continue
        else:
            Time[cnt] = (1 - time / counter)

        # 记录前5个有效点
        if cnt < 5:
            debug_thresholds.append({
                'Th': Th,
                'Precision': Precision[cnt],
                'Recall': Recall[cnt],
                'Time': Time[cnt],
                'Tp': Tp,
                'Tp_Fp': Tp_Fp
            })

        cnt += 1

    print(f"\nValid threshold points: {cnt}")

    if cnt == 0:
        print("\n⚠️  WARNING: No valid threshold points found!")
        print("   Possible reasons:")
        print("   1. All predictions are too similar (low variance)")
        print("   2. Model hasn't learned anything yet")
        print("   3. Data loading issue")
        return 0.0, 0.0, 0.0, 0.0

    if len(debug_thresholds) > 0:
        print("\nFirst 5 valid points:")
        for i, info in enumerate(debug_thresholds):
            print(f"  Point {i + 1}: Th={info['Th']:.4f}, P={info['Precision']:.4f}, "
                  f"R={info['Recall']:.4f}, Tp={info['Tp']:.0f}, Tp_Fp={info['Tp_Fp']:.0f}")

    # 截取有效部分
    Precision = Precision[:cnt]
    Recall = Recall[:cnt]
    Time = Time[:cnt]

    print(f"\nAfter sorting by Recall:")
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]

    print(f"  - Recall range: [{Recall[0]:.4f}, {Recall[-1]:.4f}]")
    print(f"  - Precision range: [{np.min(Precision):.4f}, {np.max(Precision):.4f}]")

    _, rep_index = np.unique(Recall, return_index=True)
    print(f"  - Unique Recall values: {len(rep_index)}")

    if rep_index.size <= 1:
        print("\n⚠️  WARNING: Not enough unique Recall values!")
        print("   All samples have similar Recall, curve is degenerate.")
        return 0.0, 0.0, 0.0, 0.0

    rep_index = rep_index[1:]

    if len(rep_index) == 0:
        print("\n⚠️  WARNING: After removing first index, no points remain!")
        return 0.0, 0.0, 0.0, 0.0

    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    print(f"\nFinal P-R curve:")
    print(f"  - Points: {len(new_Recall)}")
    print(f"  - Recall range: [{new_Recall[0]:.4f}, {new_Recall[-1]:.4f}]")
    print(f"  - Precision range: [{np.min(new_Precision):.4f}, {np.max(new_Precision):.4f}]")

    # 计算 AP
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)

    idx_80 = np.where(new_Recall >= 0.8)[0]
    if idx_80.size == 0:
        P_R80 = new_Precision[-1]
        TTA_R80 = sort_time[-1] * total_seconds
        print(f"  - No Recall >= 0.8, using last point")
    else:
        P_R80 = new_Precision[idx_80[0]]
        TTA_R80 = sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds

    print(f"\n{'=' * 50}")
    print(f"Final Metrics:")
    print(f"  - AP: {AP:.4f}")
    print(f"  - mTTA: {mTTA:.4f}")
    print(f"  - TTA_R80: {TTA_R80:.4f}")
    print(f"  - P_R80: {P_R80:.4f}")
    print(f"{'=' * 50}\n")

    return AP, mTTA, TTA_R80, P_R80


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80 in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}\n'.format(APvid, AP, mTTA, TTA_R80))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    pass

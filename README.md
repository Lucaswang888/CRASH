Robustness analysis experiment of CRASH model

利用PGD进行生成攻击样本。增添了attack.py脚本
代码实现了输出的一致性和稳定性，深层特征层（Hidden state）的一致性和稳定性

运行指令（注意更改文件路径）

常规Baseline训练

python main.py \
  --dataset dad \
  --feature_name vgg16 \
  --phase train \
  --output_dir ./output/normal_train \
  --epoch 50 \
  --gpus 0

  鲁棒性框架训练

  python main.py \
  --dataset dad \
  --feature_name vgg16 \
  --phase train \
  --output_dir ./output/robust_train \
  --epoch 50 \
  --robust_train \
  --pretrained_model ./output/normal_train/dad/snapshot/final_model.pth \
  --eps 0.01 \
  --steps 5 \
  --adv_weight 1.0 \
  --sim_weight 0.5 \
  --feat_weight 0.1 \
  --gpus 0


干净数据集测试

python main.py \
  --dataset dad \
  --feature_name vgg16 \
  --phase test \
  --model_file ./output/robust_train/dad/snapshot/final_model.pth \
  --gpus 0

  加入噪声的数据集测试
  
  python main.py \
  --dataset dad \
  --feature_name vgg16 \
  --phase test \
  --model_file ./output/robust_train/dad/snapshot/final_model.pth \
  --noise_std 2.0 \
  --gpus 0
  

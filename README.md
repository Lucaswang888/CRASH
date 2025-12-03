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
  --output_dir ./output/sota_robust \
  --epoch 80 \
  --batch_size 32 \
  --robust_train \
  --pretrained_model ./output/baseline/dad/snapshot/final_model.pth \
  --eps 0.02 \
  --steps 10 \
  --alpha 0.003 \
  --adv_weight 1.0 \
  --sim_weight 0.5 \
  --feat_weight 0.05 \
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


程序支持加载模型参数继续训练（注意参数保持一致）

python main.py \
  --dataset dad \
  --feature_name vgg16 \
  --phase train \
  --output_dir ./output/sota_robust \
  --epoch 80 \
  --batch_size 32 \
  --robust_train \
  --pretrained_model ./output/baseline/dad/snapshot/final_model.pth \
  --eps 0.02 \
  --steps 10 \
  --alpha 0.003 \
  --adv_weight 1.0 \
  --sim_weight 0.5 \
  --feat_weight 0.05 \
  --gpus 0 \
  --resume \
  --model_file ./output/sota_robust/dad/snapshot/model_XX.pth  <-- 这里改成你中断处的模型文件
  

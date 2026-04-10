export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# conda deactivate && source
# conda activate hfmnet

# ========== baseline 实验运行命令 ==========

python train.py --opt options/train.yaml
# python exp_tta/test_tta.py --opt exp_tta/options/test.yaml --tta
# nohup python train.py --opt options/train.yaml >>logs/baseline.log&

# ========== 5个实验运行命令 ==========

# 1. boundary_loss (device: 0)
# python train.py --opt exp_boundary_loss/options/train.yaml

# # 2. warmup_onecycle (device: 1)
# python train.py --opt exp_warmup_onecycle/options/train.yaml

# # 3. gradual_unfreeze (device: 1)
# python train.py --opt exp_gradual_unfreeze/options/train.yaml

# # 4. larger_backbone (device: 1)
# python train.py --opt exp_larger_backbone/options/train.yaml

# # 5. tta (device: 0, 仅训练)
# python train.py --opt exp_tta/options/train.yaml

# ========== 并行运行示例 (使用 nohup) ==========
# nohup python train.py --opt exp_boundary_loss/options/train.yaml >>logs/boundary_loss.log&
# nohup python train.py --opt exp_warmup_onecycle/options/train.yaml >>logs/warmup_onecycle.log&
# nohup python train.py --opt exp_gradual_unfreeze/options/train.yaml >>logs/gradual_unfreeze.log&
# nohup python train.py --opt exp_larger_backbone/options/train.yaml >>logs/larger_backbone.log&
# nohup python train.py --opt exp_tta/options/train.yaml >>logs/tta.log&




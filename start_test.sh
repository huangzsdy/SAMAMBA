export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ========== 测试命令 ==========

# Baseline
python test.py --opt options/test.yaml

# 1. boundary_loss
# python test.py --opt exp_boundary_loss/options/test.yaml

# 2. warmup_onecycle
# python test.py --opt exp_warmup_onecycle/options/test.yaml

# 3. gradual_unfreeze
# python test.py --opt exp_gradual_unfreeze/options/test.yaml

# 4. larger_backbone
# python test.py --opt exp_larger_backbone/options/test.yaml

# 5. TTA (使用专门的 test_tta.py)
# python exp_tta/test_tta.py --opt exp_tta/options/test.yaml

# ========== 并行运行示例 ==========
# nohup python test.py --opt options/test.yaml >logs/test_baseline.log&
# nohup python test.py --opt exp_boundary_loss/options/test.yaml >logs/test_boundary_loss.log&
# nohup python test.py --opt exp_warmup_onecycle/options/test.yaml >logs/test_warmup_onecycle.log&
# nohup python test.py --opt exp_gradual_unfreeze/options/test.yaml >logs/test_gradual_unfreeze.log&
# nohup python test.py --opt exp_larger_backbone/options/test.yaml >logs/test_larger_backbone.log&
# nohup python exp_tta/test_tta.py --opt exp_tta/options/test.yaml >logs/test_tta.log&

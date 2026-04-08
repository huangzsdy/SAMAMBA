export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# conda deactivate && source
# conda activate hfmnet

# python exp_tta/test_tta.py --opt exp_tta/options/test.yaml --tta

# python train.py --opt exp_tta/options/train.yaml 
# python train.py --opt exp_boundary_loss/options/train.yaml
# python train.py --opt exp_gradual_unfreeze/options/train.yaml
# python train.py --opt exp_larger_backbone/options/train.yaml
python train.py --opt exp_warmup_onecycle/options/train.yaml




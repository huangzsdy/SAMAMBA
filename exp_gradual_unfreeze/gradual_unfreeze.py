"""
渐进式解冻 (Gradual Unfreezing) 模块

训练策略:
1. 阶段1 (0-150 epoch): 冻结骨干网络，仅训练适配器和解码器
2. 阶段2 (151-300 epoch): 解冻后半部分编码器，使用较小学习率
3. 阶段3 (301-400 epoch): 完全解冻编码器，使用更小学习率微调

使用方法:
    1. 在 train.yaml 中启用:
        model:
          gradual_unfreeze:
            enabled: True
            stages: [...]

    2. 在 train.py 中导入并应用:
        from exp_gradual_unfreeze.gradual_unfreeze import GradualUnfreezer

        # 创建 GradualUnfreezer 实例
        unfreezer = GradualUnfreezer(model, opt)

        # 在每个 epoch 开始时调用
        unfreezer.update(epoch)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger('basicseg')


class GradualUnfreezer:
    """
    渐进式解冻管理器

    在训练过程中逐步解冻骨干网络的参数
    """

    def __init__(self, model, opt):
        """
        初始化解冻管理器

        Args:
            model: 网络模型 (Seg_model 或 Base_model)
            opt: 配置选项字典
        """
        self.model = model
        self.opt = opt
        self.enabled = opt.get('model', {}).get('gradual_unfreeze', {}).get('enabled', False)

        if not self.enabled:
            logger.info("Gradual unfreezing is disabled")
            return

        # 获取解冻阶段配置
        self.stages = opt.get('model', {}).get('gradual_unfreeze', {}).get('stages', [])
        self.stage2_lr_factor = opt.get('model', {}).get('gradual_unfreeze', {}).get('stage2_lr_factor', 0.1)
        self.stage3_lr_factor = opt.get('model', {}).get('gradual_unfreeze', {}).get('stage3_lr_factor', 0.01)

        # 获取当前学习率
        self.base_lr = opt.get('model', {}).get('optim', {}).get('init_lr', 5e-4)

        # 当前阶段
        self.current_stage = None

        # 记录已解冻的层
        self.unfrozen_layers = set()

        if self.enabled:
            logger.info(f"Gradual unfreezing enabled with {len(self.stages)} stages")

    def update(self, epoch):
        """
        在每个 epoch 开始时调用，更新解冻状态

        Args:
            epoch: 当前 epoch 编号
        """
        if not self.enabled:
            return

        # 找到当前阶段
        for stage in self.stages:
            if stage['start_epoch'] <= epoch <= stage['end_epoch']:
                if self.current_stage != stage['name']:
                    self.current_stage = stage['name']
                    self._apply_stage(stage, epoch)
                return

        logger.warning(f"Epoch {epoch} does not match any unfreeze stage")

    def _apply_stage(self, stage, epoch):
        """
        应用解冻阶段

        Args:
            stage: 阶段配置字典
            epoch: 当前 epoch
        """
        stage_name = stage['name']
        unfreeze_layers = stage.get('unfreeze_layers', [])

        logger.info(f"=== Applying unfreeze stage: {stage_name} (epoch {epoch}) ===")

        # 获取网络
        net = self.model.net
        if hasattr(net, 'module'):
            net = net.module

        # 解冻指定的层
        for layer_name in unfreeze_layers:
            self._unfreeze_layer(net, layer_name)

        # 调整学习率
        lr_factor = 1.0
        if stage_name == 'stage2_partial':
            lr_factor = self.stage2_lr_factor
        elif stage_name == 'stage3_full':
            lr_factor = self.stage3_lr_factor

        if lr_factor < 1.0:
            self._adjust_learning_rate(lr_factor)

        # 打印当前可训练参数信息
        self._print_trainable_params()

    def _unfreeze_layer(self, net, layer_name):
        """
        解冻指定层

        Args:
            net: 网络模型
            layer_name: 层名称 (如 'encoder.blocks.8')
        """
        parts = layer_name.split('.')

        try:
            module = net
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)

            # 解冻参数
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    self.unfrozen_layers.add(layer_name)
                    logger.info(f"Unfroze layer: {layer_name}")

        except Exception as e:
            logger.warning(f"Failed to unfreeze layer {layer_name}: {e}")

    def _adjust_learning_rate(self, lr_factor):
        """
        调整学习率

        Args:
            lr_factor: 学习率乘数
        """
        new_lr = self.base_lr * lr_factor

        # 更新优化器中的学习率
        if hasattr(self.model, 'optim') and self.model.optim is not None:
            for param_group in self.model.optim.param_groups:
                # 只更新已解冻的参数
                if param_group.get('requires_grad', True):
                    param_group['lr'] = new_lr

        logger.info(f"Adjusted learning rate to {new_lr:.2e} (factor: {lr_factor})")

    def _print_trainable_params(self):
        """打印当前可训练参数信息"""
        net = self.model.net
        if hasattr(net, 'module'):
            net = net.module

        total_params = 0
        trainable_params = 0

        for param in net.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")


class GradualUnfreezerSimple:
    """
    简化版渐进式解冻

    基于 epoch 数量自动解冻
    """

    def __init__(self, model, unfreeze_epochs=[150, 300], lr_factors=[0.1, 0.01]):
        """
        Args:
            model: 网络模型
            unfreeze_epochs: 解冻epoch列表，如 [150, 300] 表示:
                - 0-149: 全部冻结
                - 150-299: 解冻50%
                - 300+: 完全解冻
            lr_factors: 对应的学习率因子，如 [0.1, 0.01]
        """
        self.model = model
        self.unfreeze_epochs = unfreeze_epochs
        self.lr_factors = lr_factors
        self.current_stage = -1

    def update(self, epoch):
        """更新解冻状态"""
        # 确定当前阶段
        stage = -1
        for i, unfreeze_epoch in enumerate(self.unfreeze_epochs):
            if epoch >= unfreeze_epoch:
                stage = i

        # 如果阶段变化，执行解冻
        if stage != self.current_stage:
            self.current_stage = stage

            if stage == -1:
                logger.info(f"Epoch {epoch}: All layers frozen (train adapter only)")
            else:
                # 解冻更多层
                self._unfreeze_more(stage)

                # 调整学习率
                lr_factor = self.lr_factors[stage]
                self._adjust_lr(lr_factor, epoch)

    def _unfreeze_more(self, stage):
        """根据阶段解冻更多层"""
        net = self.model.net
        if hasattr(net, 'module'):
            net = net.module

        # 获取编码器块
        if hasattr(net, 'encoder') and hasattr(net.encoder, 'blocks'):
            blocks = net.encoder.blocks

            if isinstance(blocks, nn.Sequential):
                num_blocks = len(blocks)
                # 解冻比例: stage=0 -> 50%, stage=1 -> 100%
                unfreeze_ratio = (stage + 1) / 2
                unfreeze_count = int(num_blocks * unfreeze_ratio)

                logger.info(f"Unfreezing {unfreeze_count}/{num_blocks} encoder blocks")

                for i in range(unfreeze_count):
                    for param in blocks[i].parameters():
                        param.requires_grad = True

    def _adjust_lr(self, lr_factor, epoch):
        """调整学习率"""
        base_lr = 5e-4  # 基础学习率
        new_lr = base_lr * lr_factor

        if hasattr(self.model, 'optim'):
            for param_group in self.model.optim.param_groups:
                param_group['lr'] = new_lr

        logger.info(f"Epoch {epoch}: Learning rate adjusted to {new_lr:.2e}")


# 辅助函数: 冻结/解冻所有编码器
def freeze_encoder(model):
    """冻结编码器所有参数"""
    net = model.net
    if hasattr(net, 'module'):
        net = net.module

    if hasattr(net, 'encoder'):
        for param in net.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")


def unfreeze_encoder(model):
    """解冻编码器所有参数"""
    net = model.net
    if hasattr(net, 'module'):
        net = net.module

    if hasattr(net, 'encoder'):
        for param in net.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen")


# 使用示例
"""
# 在 train.py 中:

from exp_gradual_unfreeze.gradual_unfreeze import GradualUnfreezerSimple

# 初始化
unfreezer = GradualUnfreezerSimple(model, unfreeze_epochs=[150, 300], lr_factors=[0.1, 0.01])

# 在训练循环中:
for epoch in range(1, total_epochs + 1):
    # 更新解冻状态
    unfreezer.update(epoch)

    # 继续训练...
"""
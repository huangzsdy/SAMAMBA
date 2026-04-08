"""
OneCycleLR 学习率调度器支持

扩展 base_model.py 以支持 OneCycleLR

使用方法:
    在 train.yaml 中配置:

    lr:
      warmup_iter: 1000
      scheduler:
        type: OneCycleLR
        step_interval: iter
        max_lr: !!float 1e-3
        pct_start: 0.3
        anneal_strategy: cos
        div_factor: 25.0
        final_div_factor: 1000.0
"""

import torch
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
from basicseg.utils.registry import LR_SCHEDULER_REGISTRY


class OneCycleLR(TorchOneCycleLR):
    """
    OneCycleLR 学习率调度器

    特点:
    - 先快速上升学习率，然后逐渐下降
    - 在训练过程中会超过初始学习率
    - 最终学习率会非常小

    参数:
        max_lr: 最大学习率
        total_steps: 总迭代次数
        pct_start: 用于 warmup 的步数比例
        anneal_strategy: 'cos' 或 'linear'
        div_factor: 初始学习率 = max_lr / div_factor
        final_div_factor: 最终学习率 = 初始学习率 / final_div_factor
    """
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 anneal_strategy='cos', div_factor=25.0, final_div_factor=1000.0,
                 three_phase=False, last_epoch=-1):

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        # 计算三个阶段的步数
        self.steps_up = int(total_steps * pct_start)
        self.steps_down = total_steps - self.steps_up

        # 初始学习率 = max_lr / div_factor
        initial_lr = max_lr / div_factor
        # 最终学习率 = initial_lr / final_div_factor
        min_lr = initial_lr / final_div_factor

        super().__init__(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch
        )

    def get_lr(self):
        """获取当前学习率"""
        return self.get_last_lr()


# 注册到 LR_SCHEDULER_REGISTRY (如果存在)
try:
    @LR_SCHEDULER_REGISTRY.register()
    class OneCycleLR_Scheduler(OneCycleLR):
        pass
except NameError:
    pass


def create_onecycle_scheduler(optimizer, opt, total_iters):
    """
    创建 OneCycleLR 调度器

    Args:
        optimizer: 优化器
        opt: 配置字典
        total_iters: 总迭代次数

    Returns:
        OneCycleLR 调度器实例
    """
    lr_opt = opt.get('model', {}).get('lr', {})
    scheduler_opt = lr_opt.get('scheduler', {})

    max_lr = scheduler_opt.get('max_lr', 1e-3)
    pct_start = scheduler_opt.get('pct_start', 0.3)
    anneal_strategy = scheduler_opt.get('anneal_strategy', 'cos')
    div_factor = scheduler_opt.get('div_factor', 25.0)
    final_div_factor = scheduler_opt.get('final_div_factor', 1000.0)
    three_phase = scheduler_opt.get('three_phase', False)

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_iters,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase
    )


# 补丁: 修改 base_model.py 中的 setup_lr_schduler 方法
# 以下是修改后的 setup_lr_schduler 函数，可以直接替换原函数

def patched_setup_lr_schduler(self):
    """
    修复后的学习率调度器设置，支持 OneCycleLR

    需要在 base_model.py 中替换原来的 setup_lr_schduler 方法
    """
    lr_opt = self.opt['model']['lr']
    self.step_interval = lr_opt['scheduler'].pop('step_interval', 'epoch')

    if self.step_interval == 'epoch':
        self.T_max = self.opt['exp']['total_epochs']
    elif self.step_interval == 'iter':
        self.T_max = self.opt['exp']['total_iters']

    self.warmup_iter = lr_opt['warmup_iter']
    scheduler_type = lr_opt['scheduler'].pop('type')

    if scheduler_type is None:
        self.scheduler = None
    else:
        # 原有的调度器
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            from basicseg.utils.lr_scheduler import MultiStepRestartLR
            self.scheduler = MultiStepRestartLR(self.optim, **lr_opt['scheduler'])

        elif scheduler_type == 'CosineAnnealingRestartLR':
            from basicseg.utils.lr_scheduler import CosineAnnealingRestartLR
            self.scheduler = CosineAnnealingRestartLR(self.optim, **lr_opt['scheduler'])

        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, self.T_max, **lr_opt['scheduler'])

        elif scheduler_type == 'Poly':
            from basicseg.utils.lr_scheduler import PolyLR
            self.scheduler = PolyLR(self.optim, self.T_max, **lr_opt['scheduler'])

        # 新增: OneCycleLR 支持
        elif scheduler_type == 'OneCycleLR':
            # 计算总步数
            total_steps = self.T_max

            # 获取 OneCycleLR 特有参数
            max_lr = lr_opt['scheduler'].get('max_lr', 1e-3)
            pct_start = lr_opt['scheduler'].get('pct_start', 0.3)
            anneal_strategy = lr_opt['scheduler'].get('anneal_strategy', 'cos')
            div_factor = lr_opt['scheduler'].get('div_factor', 25.0)
            final_div_factor = lr_opt['scheduler'].get('final_div_factor', 1000.0)
            three_phase = lr_opt['scheduler'].get('three_phase', False)

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                three_phase=three_phase
            )

        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')


# 使用说明
"""
要使用 OneCycleLR，需要修改 basicseg/base_model.py 中的 setup_lr_schduler 方法:

1. 找到 setup_lr_schduler 方法
2. 在 scheduler_type == 'CosineAnnealingLR' 分支后添加:

elif scheduler_type == 'OneCycleLR':
    total_steps = self.T_max
    max_lr = lr_opt['scheduler'].get('max_lr', 1e-3)
    pct_start = lr_opt['scheduler'].get('pct_start', 0.3)
    anneal_strategy = lr_opt['scheduler'].get('anneal_strategy', 'cos')
    div_factor = lr_opt['scheduler'].get('div_factor', 25.0)
    final_div_factor = lr_opt['scheduler'].get('final_div_factor', 1000.0)

    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        self.optim,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
"""
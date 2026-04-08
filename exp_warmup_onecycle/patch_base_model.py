"""
补丁: 为 base_model.py 添加 OneCycleLR 支持

使用方法:
1. 备份原文件: cp basicseg/base_model.py basicseg/base_model.py.bak
2. 运行此脚本: python exp_warmup_onecycle/patch_base_model.py

这将在 base_model.py 中添加 OneCycleLR 支持
"""

import os
import re


def patch_base_model():
    """为 base_model.py 添加 OneCycleLR 支持"""
    base_model_path = 'basicseg/base_model.py'

    if not os.path.exists(base_model_path):
        print(f"错误: 找不到 {base_model_path}")
        return False

    # 读取文件
    with open(base_model_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经添加了 OneCycleLR 支持
    if 'OneCycleLR' in content:
        print("base_model.py 已经包含 OneCycleLR 支持")
        return True

    # 找到 setup_lr_schduler 方法中的 CosineAnnealingLR 分支
    # 在其后面添加 OneCycleLR 分支

    # 匹配模式: 找到 CosineAnnealingLR 分支的结束位置
    pattern = r"(elif scheduler_type == 'CosineAnnealingLR':\s+self\.scheduler = \s+torch\.optim\.lr_scheduler\.CosineAnnealingLR\(self\.optim, self\.T_max, \*\*lr_opt\['scheduler'\]\))\s+(elif scheduler_type == 'Poly':)"

    replacement = r'''\1

            # OneCycleLR 学习率调度器
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

            \2'''

    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        print("警告: 未能找到需要修改的位置")
        print("请手动添加 OneCycleLR 支持")
        return False

    # 写入修改后的文件
    with open(base_model_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("成功添加 OneCycleLR 支持到 base_model.py")
    return True


def revert_base_model():
    """恢复原始的 base_model.py"""
    base_model_path = 'basicseg/base_model.py'
    backup_path = 'basicseg/base_model.py.bak'

    if not os.path.exists(backup_path):
        print("错误: 找不到备份文件")
        return False

    # 恢复备份
    with open(backup_path, 'r', encoding='utf-8') as f:
        content = f.read()

    with open(base_model_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("成功恢复原始 base_model.py")
    return True


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'revert':
        revert_base_model()
    else:
        patch_base_model()
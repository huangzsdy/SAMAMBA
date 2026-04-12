import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as Data
import os
import time
import logging
import random
import copy
try:
    import setproctitle
except ImportError:
    setproctitle = None
import numpy as np
from basicseg.seg_model import Seg_model
from basicseg.utils.yaml_options import parse_options, dict2str
from basicseg.utils.path_utils import *
from basicseg.utils.logger import get_root_logger, init_tb_logger, get_env_info, MessageLogger
from basicseg.data import build_dataset

def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # 启用 TF32 加速 (Ampere 及以上 GPU)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def init_exp(opt, args):
    exp_name = opt['exp'].get('name')
    if not exp_name:
        exp_name = os.path.basename(args.opt[:-4])
        opt['exp']['name'] = exp_name
    exp_root = make_exp_root(os.path.join('experiment', exp_name))
    opt['exp']['exp_root'] = exp_root
    log_file = os.path.join(exp_root, f'train_{exp_name}_{get_time_str()}.log')
    logger = get_root_logger(logger_name='basicseg', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_logger(log_dir = os.path.join(exp_root, 'tb_log'))
    return logger, tb_logger

def init_model(opt):

    model = Seg_model(opt)
    return model

def init_dataset(opt):
    # trainset
    train_opt = opt['dataset']['train']
    trainset = build_dataset(train_opt)
    test_opt = opt['dataset']['test']
    testset = build_dataset(test_opt)
    return trainset, testset

def init_dataloader(opt, trainset, testset):
    num_workers = opt['exp'].get('nw', 0)  # default 0 for stability
    prefetch_factor = opt['exp'].get('prefetch_factor', 2)  # 预取因子
    pin_memory = opt['exp'].get('pin_memory', True)  # 锁页内存加速传输

    if opt['exp']['dist']:
        sampler = Data.DistributedSampler(trainset)
    else:
        sampler = None

    train_loader = Data.DataLoader(dataset=trainset, batch_size=opt['exp']['bs'],\
                                    sampler=sampler, num_workers=num_workers,
                                    persistent_workers=False if num_workers == 0 else True,
                                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                                    pin_memory=pin_memory)
    test_loader  = Data.DataLoader(dataset=testset, batch_size=opt['exp']['bs'],\
                                    sampler=None, num_workers=num_workers,
                                    persistent_workers=False if num_workers == 0 else True,
                                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                                    pin_memory=pin_memory)
    return train_loader, test_loader

def main():
    opt, args = parse_options()

    import setproctitle
    setproctitle.setproctitle(opt['exp']['name'])

    # 自动检测单卡还是多卡模式
    # 方式1: 使用 torch.distributed.launch (设置 WORLD_SIZE)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = args.local_rank if args.local_rank >= 0 else 0

    if world_size > 1:
        # 多卡模式 (通过 torch.distributed.launch 启动)
        opt['exp']['dist'] = True
        dist.init_process_group(backend='nccl')
        total_device = world_size
        opt['exp']['num_devices'] = total_device
        cur_rank = dist.get_rank()
        torch.cuda.set_device(cur_rank)
    else:
        # 单卡模式 - 检查是否在 yaml 中配置了多卡字符串
        if isinstance(opt['exp']['device'], str) and ',' in opt['exp']['device']:
            # 多卡字符串 "0,1,2,3" - 不支持这种方式，改用 launch 或 spawn
            print("Warning: 多卡字符串格式不支持，请使用 torch.distributed.launch 方式启动")
            print("例如: python -m torch.distributed.launch --nproc_per_node=4 train.py --opt options/train.yaml")
            opt['exp']['dist'] = False
            cur_rank = 0
            total_device = 1
            opt['exp']['num_devices'] = total_device
        else:
            # 单卡模式
            if isinstance(opt['exp']['device'], int):
                os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['exp']['device'])
            elif isinstance(opt['exp']['device'], str):
                os.environ['CUDA_VISIBLE_DEVICES'] = opt['exp']['device']
            opt['exp']['dist'] = False
            cur_rank = 0
            total_device = 1
            opt['exp']['num_devices'] = total_device

    # init dataset
    trainset, testset = init_dataset(opt)
    train_loader, test_loader = init_dataloader(opt, trainset, testset)

    # init exp_root, logger, tb_logger
    total_epochs = opt['exp']['total_epochs']
    total_iters = total_epochs * (len(trainset) // opt['exp']['bs'] // total_device +1)
    opt['exp']['total_iters'] = total_iters
    save_interval = opt['exp']['save_interval']
    test_interval = opt['exp']['test_interval']
    logger, tb_logger = init_exp(opt, args)
    set_seed(cur_rank + 0)
    # 初始化 模型参数, 包含 网络 优化器 损失函数 学习率准则
    # initialize parameters including network, optimizer, loss function, learning rate scheduler
    model = init_model(opt)
    # model.load_from() #vmunet
    cur_iter = 0
    cur_epoch = 1
    # 从断点继续训练
    # train from checkpoint
    if opt.get('resume'):
        if opt['resume'].get('net_path'):
            model.load_network(model.net, opt['resume']['net_path'], strict=False)
            logger.info(f'load pretrained network from: {opt["resume"]["net_path"]}')
        else:
            logger.info(f'load from random initialized network')
        if opt['resume'].get('state_path'):
            cur_epoch = model.resume_training(opt['resume']['state_path'])
            cur_iter = cur_epoch * (len(trainset) // opt['exp']['bs'] // total_device + 1)
            logger.info(f'resume training from epoch: {cur_epoch}')
        else:
            logger.info(f'training from epoch: 1')

    msg_logger = MessageLogger(opt, start_epoch=cur_epoch, tb_logger=tb_logger)
    logger.info(f'Start training... Total epochs: {total_epochs}, Total iters: {total_iters}')
    for epoch in range(cur_epoch, total_epochs+1):
        if opt['exp']['dist']:
            train_loader.sampler.set_epoch(epoch)
        epoch_st_time = time.time()
        logger.info(f'Start epoch {epoch}/{total_epochs}')
        ########## training ##########
        for idx, data in enumerate(train_loader):
            cur_iter += 1
            model.update_learning_rate(cur_iter, idx)
            model.optimize_one_iter(data, epoch)

            # 打印每个 batch 的 loss
            if cur_iter % opt['exp'].get('log_interval', 10) == 0:
                batch_loss_dict = model.get_batch_loss(reduction='sum')
                loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in batch_loss_dict.items()])
                logger.info(f'Epoch {epoch}, Iter {cur_iter}, Batch {idx}, Loss: {loss_str}')

        epoch_time = time.time() - epoch_st_time
        log_vars = {'epoch': epoch}
        log_vars.update({'lrs': model.get_current_learning_rate()})
        log_vars.update({'time': epoch_time})
        log_vars.update({'train_loss': model.get_epoch_loss(opt['exp']['dist'], 'sum')})
        log_vars.update({'train_mean_metric': model.get_mean_metric(opt['exp']['dist'], 'mean')})
        log_vars.update({'train_norm_metric': model.get_norm_metric(opt['exp']['dist'], 'mean')})
        ########## tesing ##########
        if cur_rank == 0 and epoch % test_interval == 0:
            # model.net.eval()
            model.model_to_eval()
            model.reset_metric()  # 重置 metric，确保重新计算
            for idx, data in enumerate(test_loader):
                model.test_one_iter(data)
            log_vars.update({'test_loss': model.get_epoch_loss()})
            test_mean_metric = model.get_mean_metric()
            print(f"DEBUG after get_mean_metric: tp={model.metric.tp}, fp={model.metric.fp}, fn={model.metric.fn}")
            test_norm_metric = model.get_norm_metric()
            print(f"DEBUG test_mean_metric: {test_mean_metric}")
            print(f"DEBUG test_norm_metric: {test_norm_metric}")
            log_vars.update({'test_mean_metric': test_mean_metric})
            log_vars.update({'test_norm_metric': test_norm_metric})
            if test_mean_metric['iou'] > model.best_mean_metric['iou']:
                model.best_mean_metric['iou'] = test_mean_metric['iou']
                model.best_mean_metric['net'] = copy.deepcopy(model.net.state_dict())
                model.best_mean_metric['epoch'] = epoch
            if test_norm_metric['iou'] > model.best_norm_metric['iou']:
                model.best_norm_metric['iou'] = test_norm_metric['iou']
                model.best_norm_metric['net'] = copy.deepcopy(model.net.state_dict())
                model.best_norm_metric['epoch'] = epoch
            if test_mean_metric['fscore'] > model.best_F1['F1']:
                model.best_F1['F1'] = test_mean_metric['fscore']
                model.best_F1['net'] = copy.deepcopy(model.net.state_dict())
                model.best_F1['epoch'] = epoch
            # model.net.train()
            model.model_to_train()
        ########## saving_model ##########
        if cur_rank == 0 and epoch % save_interval == 0 :
            model.save_network(opt, model.net, epoch)
            model.save_training_state(opt, epoch)

        msg_logger(log_vars)

    ########## trainging done ##########
    if cur_rank == 0:
        model.save_network(opt, model.net, current_epoch='latest')
        if model.best_mean_metric['net'] is not None:
            model.save_network(opt, model.best_mean_metric['net'], current_epoch='best_mean', net_dict=True)
        if model.best_norm_metric['net'] is not None:
            model.save_network(opt, model.best_norm_metric['net'], current_epoch='best_norm', net_dict=True)
        if model.best_F1['net'] is not None:
            model.save_network(opt, model.best_F1['net'], current_epoch='best_F1', net_dict=True)
        logger.info(f"best_mean_metric: [epoch: {model.best_mean_metric['epoch']}] [iou: {model.best_mean_metric['iou']:.4f}]")
        logger.info(f"best_norm_metric: [epoch: {model.best_norm_metric['epoch']}] [iou: {model.best_norm_metric['iou']:.4f}]")
        logger.info(f"best_F1: [epoch: {model.best_F1['epoch']}] [F1: {model.best_F1['F1']:.4f}]")

if __name__ == '__main__':
    main()
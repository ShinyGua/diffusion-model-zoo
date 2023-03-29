import datetime
import shutil
import time
import warnings
from pathlib import Path

import math
import torch
import torch.distributed as dist
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast

from data import build_loader
from model import build_ddpm, EMAHelper
from utils import create_scheduler, optimizer_kwargs, save_checkpoint, set_weight_decay, multi_process_setup, \
    distributed_training, auto_resume, generate_xt, backward, get_alpha_and_beta, check_path_is_file_or_dir, \
    get_img_from_dataloader_iter

warnings.filterwarnings("ignore")


def main(local_rank):
    config, logger = multi_process_setup(local_rank)

    # 创建 datasets, dataset_loader
    # build datasets, dataset_loader
    dsets, dset_loaders = build_loader(config)

    # 创建模型
    # build the model
    model = build_ddpm(config, logger, with_adapter=(not config.model.ddpm.pretrain))
    model.cuda()

    scaler = GradScaler()
    if dist.get_rank() == 0:
        logger.info("==============>ddpm_model....................")
    logger.info(str(model))

    # 创建优化器
    # build the optimizer
    params_list = set_weight_decay(model)
    optimizer = create_optimizer_v2(params_list, **optimizer_kwargs(config))

    # 学习率调整器
    # learning rate scheduler
    lr_scheduler, num_iteration = create_scheduler(config, optimizer)

    start_iteration = 0

    # 多卡训练设置
    # distributed training
    models, optimizer = distributed_training([model], optimizer, local_rank, logger)
    model = models[0]
    model_without_ddp = model.module

    if config.model.ddpm.ema:
        ema_helper = EMAHelper(mu=config.model.ddpm.ema_rate)
        ema_helper.register(model)
    else:
        ema_helper = None

    if config.auto_resume and check_path_is_file_or_dir(Path(config.output).joinpath("latest.pt")):
        start_iteration = auto_resume(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger, ema_helper,
                                      "latest.pt")

    if ema_helper:
        ema_helper.to_cuda(model)
    lr_scheduler.step(start_iteration)
    dset_loaders["train"].sampler.set_epoch(start_iteration)

    s_time = time.time()
    logger.info(f"==============>Start train model....................")
    training(model, dset_loaders, optimizer, lr_scheduler, ema_helper, scaler, logger, config,
             start_iteration)

    end_time = time.time() - s_time
    logger.info(f"Training takes {datetime.timedelta(seconds=int(end_time))}")
    logger.info("Done!")


def training(model, dset_loaders, optimizer, lr_scheduler, ema_helper, scaler, logger, config, start_iteration=0):
    model.train()
    model_without_ddp = model.module

    optimizer.zero_grad()

    data_iter = iter(dset_loaders['train'])
    iterations = config.train.iteration

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    num_timesteps = config.dm.num_diffusion_timesteps
    a, betas = get_alpha_and_beta(config)

    end = time.time()
    criterion = torch.nn.MSELoss(reduction='none').cuda()

    # use the half precision or normal train
    dtype = torch.float16 if config.float16 else torch.float32

    with autocast(dtype=dtype):
        for idx in range(start_iteration, iterations):
            x0 = get_img_from_dataloader_iter(dset_loaders['train'], data_iter, idx)

            # reset each meter
            if idx % config.reset_average_meter == 0:
                batch_time.reset()
                data_time.reset()
                loss_meter.reset()
                norm_meter.reset()
                logger.info("\t")

            _B = x0.shape[0]
            # get the noised xt from x0 at t as DDPM
            e, xt, t = generate_xt(x0, a, _B, num_timesteps)

            # measure data loading time
            data_time.update(time.time() - end)

            output = model(xt, t.float())
            loss = criterion(e, output).sum(dim=[1, 2, 3]).mean(dim=0)

            grad_norm = backward(model, loss, optimizer, scaler, config)

            if ema_helper is not None:
                ema_helper.update(model)

            torch.cuda.synchronize()
            # record loss
            loss_meter.update(loss.item(), _B)
            if not math.isnan(grad_norm) and not math.isinf(grad_norm):
                norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            del loss, xt, e, output

            lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step_update(num_updates=idx + 1, metric=loss_meter.avg)
            lr_scheduler.step(idx + 1)

            if (idx % 100 == 0 or idx == (iterations - 1)):
                save_checkpoint(config=config,
                                model=model_without_ddp,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                iteration=idx + 1,
                                scaler=scaler,
                                logger=logger,
                                ema=ema_helper,
                                name=f"latest.pt")

            # save latest checkpoint each 100
            if (idx % config.save_freq == 0 or idx == (iterations - 1)):
                if dist.get_rank() == 0:
                    shutil.copytree(Path(config.output).joinpath(f"latest.pt"),
                                    Path(config.output).joinpath(f"iteration_{idx}.pt"))

            # copy checkpoint each log_freq
            if idx % config.log_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (iterations - idx)
                logger.info(
                    f'Train: [{idx}/{iterations}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                    f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

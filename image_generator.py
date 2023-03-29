import datetime
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as tvu
from timm.optim import create_optimizer_v2
from torch.cuda.amp import autocast

from model import build_ddpm
from utils import compute_alpha, get_named_beta_schedule, multi_process_setup, distributed_training, set_weight_decay, \
    optimizer_kwargs


def main(local_rank):
    config, logger = multi_process_setup(local_rank)
    gpus = torch.cuda.device_count()

    dataset_len = 64
    if config.max_length != None:
        dataset_len = config.max_length

    # build the ddpm model
    ddpm_model = build_ddpm(config, logger, True)
    ddpm_model.cuda()
    ddpm_model.eval()

    params_list = set_weight_decay(ddpm_model)
    optimizer = create_optimizer_v2(params_list, **optimizer_kwargs(config))

    # 混合精度以及多卡训练设置
    # Mixed-Precision and distributed training
    models, optimizer = distributed_training([ddpm_model], optimizer, local_rank, logger)
    ddpm_model = models[0]

    timesteps = config.dm.num_diffusion_timesteps

    betas = torch.from_numpy(get_named_beta_schedule(schedule_name=config.dm.schedule_name,
                                                     num_diffusion_timesteps=timesteps,
                                                     beta_start=config.dm.beta_start,
                                                     beta_end=config.dm.beta_end)).float()
    betas = betas.cuda(non_blocking=True)

    eta = config.dm.eta
    skip = int(config.dm.num_diffusion_timesteps / config.dm.sample_timesteps)

    if config.dm.skip_type == "uniform":
        seq = range(0, timesteps, skip)
    elif config.dm.skip_type == "quad":
        seq = (
                np.linspace(
                    0, np.sqrt(timesteps * 0.8), config.dm.sample_timesteps
                )
                ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError

    B = config.model.batch_size

    num = 0

    out_dir_path = Path(config.output)
    out_path = out_dir_path.joinpath("fake_imgs")
    out_path.mkdir(parents=True, exist_ok=True)

    # count the saved image number
    saved_num = len([file for file in out_path.glob(f'**/*.png')])

    idx = 0

    while num < dataset_len:
        with autocast(dtype=torch.float16):
            s_time = time.time()
            x = torch.randn(B,
                            3,
                            config.data.img_size,
                            config.data.img_size).cuda()
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xt = torch.clone(x).to(x.device)
            b = betas

            with torch.no_grad():
                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    noise = torch.randn_like(x)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())

                    # jump the saved image
                    if num < saved_num:
                        continue
                    et = ddpm_model(xt, t.float())

                    if eta != 2.0:
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_t = torch.clip(x0_t, -1.0, 1.0)
                        c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    else:
                        beta_t = 1 - at / at_next
                        mask = 1 - (t == 0).float()
                        mask = mask.view(-1, 1, 1, 1)
                        logvar = beta_t.log()
                        x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * noise
                        x0_from_e = torch.clamp(x0_from_e, -1, 1)
                        mean_eps = ((at_next.sqrt() * beta_t) * x0_from_e +
                                    ((1 - beta_t).sqrt() * (1 - at_next)) * x) / (1.0 - at)
                        mean = mean_eps
                        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise

                    xt = torch.clone(xt_next).to(x.device)
                    torch.cuda.synchronize()

            # jump the saved image
            if num < saved_num:
                num = min(num + B * gpus, saved_num)
                continue

            # save images
            imgs = xt.detach().cpu()
            imgs = (imgs + 1.0) / 2.0
            for i in range(imgs.shape[0]):
                img_path = out_path.joinpath(f"{num + i + B * int(config.local_rank)}.png")
                tvu.save_image(imgs[i,], img_path)
            idx += 1
            num += B * gpus
            torch.cuda.synchronize()
            end_time = time.time() - s_time
            logger.info(f"the number of saved image: {num}\t."
                        f'It takes {datetime.timedelta(seconds=int(end_time))}.')

    shutil.make_archive(out_dir_path.joinpath("fake_imgs").__str__(), 'zip', out_path)
    time.sleep(2)

    logger.info("Done!")


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

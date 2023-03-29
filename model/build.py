import torch
from torch.nn import Conv1d

from model.guided_diffusion import EncoderUNetModel
from utils import get_obj_from_str


def build_ddpm(config, logger, with_adapter=True):
    if "DDPM.diffusion.Model" in config.model.ddpm.target:
        # use the original DDPM from https://github.com/pesser/pytorch_diffusion/tree/master/pytorch_diffusion
        model_kwargs = {"ch": config.model.ddpm.ch,
                        "in_channels": config.model.ddpm.in_channels,
                        "out_ch": config.model.ddpm.out_ch,
                        "ch_mult": tuple(config.model.ddpm.ch_mult),
                        "num_res_blocks": config.model.ddpm.num_res_blocks,
                        "attn_resolutions": config.model.ddpm.attn_resolutions,
                        "dropout": config.model.ddpm.dropout,
                        "resamp_with_conv": config.model.ddpm.resamp_with_conv,
                        "model_type": config.model.ddpm.var_type,
                        "img_size": config.data.img_size,
                        "num_timesteps": config.dm.num_diffusion_timesteps,
                        "with_adapter": with_adapter,
                        "adapter_dim": config.model.adapter.dim,
                        "adapter_patch_size": config.model.adapter.patch_size,
                        "adapter_num_heads": config.model.adapter.num_heads,
                        "adapter_qkv_bias": config.model.adapter.qkv_bias,
                        "adapter_drop": config.model.adapter.drop}
    elif "guided_diffusion.unet.UNetModel" in config.model.ddpm.target:
        # use the guided diffusion DDPM from
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
        attention_ds = []
        for res in config.model.ddpm.attn_resolutions:
            attention_ds.append(config.data.img_size // int(res))

        model_kwargs = {"image_size": config.data.img_size,
                        "in_channels": config.model.ddpm.in_channels,
                        "model_channels": config.model.ddpm.ch,
                        "out_channels": config.model.ddpm.out_ch,
                        "num_res_blocks": config.model.ddpm.num_res_blocks,
                        "attention_resolutions": attention_ds,
                        "dropout": config.model.ddpm.dropout,
                        "channel_mult": tuple(config.model.ddpm.ch_mult),
                        "num_classes": None,
                        "use_checkpoint": True,
                        "use_fp16": config.amp_opt_level != "O0",
                        "num_heads": config.model.ddpm.num_heads,
                        "num_head_channels": config.model.ddpm.num_head_channels,
                        "num_heads_upsample": -1,
                        "use_scale_shift_norm": True,
                        "resblock_updown": True,
                        "use_new_attention_order": False,
                        "with_adapter": with_adapter,
                        "adapter_dim": config.model.adapter.dim,
                        "adapter_patch_size": config.model.adapter.patch_size,
                        "adapter_num_heads": config.model.adapter.num_heads,
                        "adapter_qkv_bias": config.model.adapter.qkv_bias,
                        "adapter_drop": config.model.adapter.drop}
    else:
        raise Exception(f"It not supports the model {config.model.ddpm.target}!")

    model = get_obj_from_str(config.model.ddpm.target)(**model_kwargs)

    if config.model.ddpm.initial_checkpoint != "":
        load_initial_checkpoint(model, config.model.ddpm.initial_checkpoint, logger)

    return model


def build_classifier(config, logger=None):
    # use the guided diffusion classifier from
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
    attention_ds = []
    for res in config.model.classifier.attn_resolutions:
        attention_ds.append(config.data.img_size // int(res))

    cfg = dict(
        image_size=config.data.img_size,
        in_channels=config.model.classifier.in_channels,
        model_channels=config.model.classifier.ch,
        out_channels=1000,
        num_res_blocks=config.model.classifier.num_res_blocks,
        attention_resolutions=attention_ds,
        channel_mult=tuple(config.model.classifier.ch_mult),
        num_head_channels=64,
        use_scale_shift_norm=True,  # False
        resblock_updown=True,  # False
        pool="attention",
        use_checkpoint=True
    )

    model = EncoderUNetModel(**cfg)

    if config.model.classifier.initial_checkpoint != "":
        try:
            load_initial_checkpoint(model, config.model.classifier.initial_checkpoint, logger)
            model.out[2].c_proj = Conv1d(model.out[2].c_proj.in_channels, 2, 1)
        except:
            model.out[2].c_proj = Conv1d(model.out[2].c_proj.in_channels, 2, 1)
            load_initial_checkpoint(model, config.model.classifier.initial_checkpoint, logger)

    return model


def load_initial_checkpoint(model, initial_checkpoint, logger):
    logger.info(f"======> Loading pre-trained model {initial_checkpoint}")
    msg = model.load_state_dict(torch.load(initial_checkpoint), strict=False)
    logger.info(msg)
    logger.info(f"======> Success loading pre-trained model {initial_checkpoint}")

from timm.scheduler import CosineLRScheduler, TanhLRScheduler, StepLRScheduler


def create_scheduler(config, optimizer):
    num_iteration = config.train.iteration

    if getattr(config.train, 'lr_noise', None) is not None:
        lr_noise = getattr(config.train, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_iteration for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_iteration
    else:
        noise_range = None

    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(config.train, 'lr_noise_pct', 0.67),
        noise_std=getattr(config.train, 'lr_noise_std', 1.),
        noise_seed=getattr(config, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(config.train, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(config.train, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(config.train, 'lr_cycle_limit', 1),
    )

    lr_min = config.train.lr * config.train.min_lr_rate
    warmup_lr_init = config.train.lr * config.train.warmup_lr_rate

    iteration = config.train.iteration
    warmup_iteration = config.train.iteration * config.train.warmup_iteration_rate

    if config.train.sched  == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=iteration,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_iteration,
            k_decay=getattr(config.train, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + config.train.cooldown_epochs
    elif config.train.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=iteration,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_iteration,
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + config.train.cooldown_epochs
    elif config.train.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.train.decay_epochs,
            decay_rate=config.train.decay_rate,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_iteration,
            **noise_args,
        )
    else:
        raise Exception(f"{config.train.sched} is not supported now!")

    return lr_scheduler, num_epochs


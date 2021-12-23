from torch.optim.lr_scheduler import MultiStepLR
from . import config

class LWarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, max_iter, milestones, gamma, pct_start, warmup_factor, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * max_iter)
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            current_lr = super().get_lr()
        return current_lr


# LinearWarmMultistep_lr
def lwarmmultistep_lr(optimizer):
    scheduler = LWarmupMultiStepLR(optimizer, max_iter=config.lwarmmultistep_T, milestones=config.lwarmmultistep_mil, gamma=config.lwarmmultistep_gam,
                                  pct_start=config.lwarmmultistep_pct, warmup_factor=config.lwarmmultistep_fac)
    return scheduler
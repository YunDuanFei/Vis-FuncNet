from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
from . import config

class CWarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, pct_start, alph, last_epoch=-1):
        self.warmup_iters = int(pct_start * milestones[0])
        self.milestones = Counter(milestones)
        self.milestones_first = milestones[0]
        self.gamma = gamma
        self.alph = alph
        super(CWarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [base_lr*self.alph for base_lr in self.base_lrs]
        elif self.warmup_iters < self.last_epoch <= self.milestones_first:
            return [base_lr for base_lr in self.base_lrs]
        else:
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


# CWarmMultistep_lr
def cwarmmultistep_lr(optimizer):
    scheduler = CWarmupMultiStepLR(optimizer, milestones=config.cwarmmultistep_mil, gamma=config.cwarmmultistep_gam,
                                  pct_start=config.cwarmmultistep_pct, alph=config.cwarmmultistep_alp)
    return scheduler
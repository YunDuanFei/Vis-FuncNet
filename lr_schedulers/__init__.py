# Constant
from .Constant import constant_lr


# Piecewise 
from .StepLR import step_lr
from .MultiStepLR import multistep_lr
from .Linear import linear_lr
from .BaseLoss import baseloss_lr


# Smooth 
from .Polynomial import polychen_lr, polybottou_lr
from .Exponential import expoloffe_lr, exporaffel_lr
from .Inverse_time import inversetime_lr
from .Linear_cosine import linearcosine_lr, linearcosinenoise_lr
from .Cosine import cosine_lr
from .Cosine_power import cosinepower_lr

# Cyclical 
from .Triangular import triangular_lr, triangular2_lr, triangularexp_lr
from .CosineClc import cosineclc_lr
from .SGDR import sgdres_lr, sgdrnes_lr, sgdrd_lr
from .Triangle import triangle_lr
from .Sine import sine_lr, sine2_lr, sineexp_lr
from .Ssgdr import ssgdr_lr, ssgdrd_lr

# Warmup (constant and linear)
from .CWarm_multistep import cwarmmultistep_lr
from .LWarm_multistep import lwarmmultistep_lr
from .LWarm_inverse_sr import lwarminversesr_lr
from .LWarm_slanted_tr import lwarmslanted_tr_lr
from .LWarm_long_trap import lwarmlong_trap_lr
from .One_cycle import lwarm1cycle_lr
from .LWarm_cosine import lwarmcosine_lr
from .LWarm_poly import lwarmpoly_lr
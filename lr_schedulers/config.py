#############################################################
#                       Constant
#############################################################
# constant
constant_size = 500
constant_gamma = 1


#############################################################
#                       Piecewise
#############################################################
# StepLR
step_size = 5
step_gamma = 0.95

# MultiStepLR 
# 500=[100, 300, 400] 1000=[200, 600, 800] 5000=[1000, 3000, 4000]
multistep_milestones = [100, 300, 400]
multistep_gamma = 0.1

# LinearLR
linear_size = 500
linear_low_lr = 0.0001

# BaseLossLR
baseloss_factor = 0.7
baseloss_patience = 10
baseloss_cooldown = 4
baseloss_min = 0.0001


#############################################################
#                        Smooth
#############################################################
# PolychenLR and PolybottouLR
polychen_T = 500
polychen_alph = 0.9
polybottou_nabda = 5
polybottou_alph = -2

# ExpoloffeLR and ExporaffelLR
expoloffe_alph = 0.009
exporaffel_T = 50

# InversetimeLR
inversetime_nabda = 2

# LinearcosineLR and with noise
linearcosine_T = 500
linearcosine_noise_decay = 20

# CosineLR
cosine_T = 500

# CosinepowerLR 
cosinepower_p = 20
cosinepower_T = 500


#############################################################
#                        Cyclical
#############################################################
# TriangularLR
triangular_base_lr = 0.0001
triangular_max_lr = 0.1
triangular_size = 50
triangular_mode = 'triangular'
triangular2_mode = 'triangular2'
triangularexp_mode = 'exp_range'
triangularexp_gamma = 0.995

# CosineClcLR
cosineclc_T = 50

# SGDRLR (not) Equally spaced
sgdres_T = 100
sgdres_Tmul = 1
sgdrnes_T = 50
sgdrnes_Tmul = 2
sgdrd_T = 500
sgdrd_num = 5
sgdrd_alph = 0.8

# TriangleLR
triangle_multi = [100, 200, 400]
triangle_gamma = 2
triangle_clen = 20

# SineLR and Sine2LR and SineExpLR
sine_size = 20
sine_nabda = 0.98

# SSGDRLR (with) decay
ssgdr_T = 500
ssgdr_size = 100
ssgdr_nabda = 0.5



#############################################################
#                        Warmup
#############################################################
# WarmMultistep_lr
cwarmmultistep_mil = [100, 250, 450]
cwarmmultistep_gam = 0.1
cwarmmultistep_pct = 0.3
cwarmmultistep_alp = 0.01

# LinearWarmMultistep_lr
lwarmmultistep_T = 500
lwarmmultistep_mil = [150, 300, 400]
lwarmmultistep_gam = 0.1
lwarmmultistep_pct = 0.3
lwarmmultistep_fac = 0.5

# LinearWarmInversesr_lr
lwarminversesr_T = 500
lwarminversesr_pct = 0.1
lwarminversesr_d = 512
lwarminversesr_muiti = 10

# LWarmupSlantedTriangularLR
lwarmslanted_tr_T = 500
lwarmslanted_tr_cut = 0.1
lwarmslanted_tr_ratio = 10

# LWarmupLongTrapezoidLR
lwarmlong_trap_T = 500
lwarmlong_trap_up = 0.2
lwarmlong_trap_down = 0.2

# LWarmup1cycleLR
lwarm1cycle_T = 500
lwarm1cycle_prcnt = 10
lwarm1cycle_div = 10

# LWarmupCosineLR
lwarmcosine_max = 500
lwarmcosine_pct = 0.2
lwarmcosine_warm = 1.0 / 3

# LWarmupPolyLR
lwarmpoly_max = 500
lwarmpoly_pct = 0.2
lwarmpoly_warm = 1.0 / 4
lwarmpoly_pow = 0.7

from detrex.config import get_config
from .models.dino_swin_tiny_224 import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
#train.init_checkpoint = "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
train.init_checkpoint ="/opt/ml/level2_objectdetection_cv-level2-cv-11/baseline/detrex/output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune/model_0089999.pth"
train.output_dir = "./output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune_pseudo"

# max training iterations
train.max_iter = 90000

# run evaluation every 5000 iters
train.eval_period = 1000

# log training infomation every 20 iters
train.log_period = 100

# save checkpoint every 5000 iters
train.checkpointer.period = 1000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-5
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

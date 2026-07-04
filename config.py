import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    def __init__(self, device="cuda:0"):
        # ---- data ----
        self.dataset = "rarp"
        self.clslist = os.path.join(REPO_ROOT, "SEDCLIP/prompt/cls.txt")

        # ---- general ----
        self.device = device
        self.seed = 2025
        self.workers = 4

        # ---- clip / features ----
        self.clip_backbone = "ViT-B/16"
        # Optional local checkpoint override. If empty, clip.load downloads
        # the official CLIP weights and caches them under ~/.cache/clip.
        self.clip_weights = ""
        self.clip_download_root = None
        self.input_size = 224
        self.feat_dim = 512
        self.num_classes = 7        
        self.ignore_last_atten = False
        self.prompt_template = "a frame of a "

        # ---- windows / batching ----
        self.frame_lenth = 32
        self.train_stride = self.frame_lenth // 2   # 16
        self.train_bs = 4

        # ---- optimisation ----
        self.lr = 1e-4

        # ---- GTA ----
        self.temporal = True
        self.head_num = 64
        self.lambda_gta = self.frame_lenth ** 2

        # ---- output ----
        self.model_name = "sedclip_baseline_GTA"
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "ckpt", self.model_name)
        self.logs_dir = self.save_dir + ".log"

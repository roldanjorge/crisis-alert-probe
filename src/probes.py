import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens.utils import get_device


class Probe(nn.Module):
    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.learning_rate = 1e-3
        self.weight_decay = 0.1
        self.betas = (0.9, 0.95)
        self.input_dim = 5120
        self.num_classes = num_classes
        self.device = get_device()
        linear = nn.Linear(self.input_dim, self.num_classes)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(linear.bias)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.num_classes), nn.Sigmoid()
        )
        self.to(self.device)

    def forward(self, resid):
        logits = self.classifier(resid)
        return logits, None

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=0
        )
        return optimizer, scheduler

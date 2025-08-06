import torch
import copy

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.ema_model.state_dict()

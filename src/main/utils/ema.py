import torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k,p in model.state_dict().items()
                       if p.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k,p in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def store(self, model): self.backup = {k: p.detach().clone() for k,p in model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self, model): model.load_state_dict({**model.state_dict(), **self.shadow})
    @torch.no_grad()
    def restore(self, model): model.load_state_dict(self.backup)


def _save_with_ema(model, path, ema):
    ema.store(model); ema.copy_to(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)
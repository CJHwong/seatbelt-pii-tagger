"""An exponential moving average of a model's parameters, for --ema."""

import torch


class WeightAverage:
    """Keeps average = decay * average + (1 - decay) * weights after every update. The
    last weights of a run at a high learning rate are noisy; the average smooths them."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.names = [name for name, _ in model.named_parameters()]
        self.values = [parameter.detach().clone() for parameter in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        torch._foreach_lerp_(self.values, [p.detach() for p in model.parameters()], 1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict:
        """The model's state with the averaged parameters in place of its own."""
        return model.state_dict() | {name: value.clone() for name, value in zip(self.names, self.values, strict=True)}

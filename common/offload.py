"""CPU-resident AdamW state with FP32 model parameters kept on their device."""
import torch


class CPUOffloadedAdamW(torch.optim.AdamW):
    """Run ordinary PyTorch AdamW on host copies after device-side AMP/clipping.

    Public parameter groups still contain the model parameters, so GradScaler
    checks their actual gradients. State dictionaries use the normal AdamW schema.
    This trades host RAM and transfers for lower GPU memory; it is not quantized.
    """

    def __init__(self, params, **kwargs):
        if kwargs.get("capturable") or kwargs.get("fused") or kwargs.get("differentiable"):
            raise ValueError("CPU offload requires noncapturable, nonfused AdamW")
        super().__init__(params, **kwargs)
        groups = []
        self._pairs = []
        for group in self.param_groups:
            hosts = []
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("CPU offload requires FP32 master parameters")
                host = torch.nn.Parameter(parameter.detach().to("cpu", copy=True))
                self._pairs.append((parameter, host))
                hosts.append(host)
            groups.append({**group, "params": hosts, "foreach": False, "fused": False,
                           "capturable": False, "differentiable": False})
        self._host_optimizer = torch.optim.AdamW(groups)

    @property
    def host_parameter_bytes(self):
        return sum(host.numel() * host.element_size() for _, host in self._pairs)

    def _sync_state(self):
        for parameter, host in self._pairs:
            self.state[parameter] = self._host_optimizer.state[host]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter, host in self._pairs:
            # Also handles loading model weights after optimizer construction.
            host.copy_(parameter.detach())
            if parameter.grad is None:
                host.grad = None
            elif host.grad is None:
                host.grad = parameter.grad.detach().to("cpu", copy=True)
            else:
                host.grad.copy_(parameter.grad.detach())
        for device_group, host_group in zip(self.param_groups, self._host_optimizer.param_groups):
            host_group.update({key: value for key, value in device_group.items() if key != "params"})
            host_group.update(foreach=False, fused=False, capturable=False, differentiable=False)
        self._host_optimizer.step()
        for parameter, host in self._pairs:
            parameter.copy_(host)
        self._sync_state()
        return loss

    def load_state_dict(self, state_dict):
        self._host_optimizer.load_state_dict(state_dict)
        for device_group, host_group in zip(self.param_groups, self._host_optimizer.param_groups):
            device_group.update({key: value for key, value in host_group.items() if key != "params"})
        self._sync_state()

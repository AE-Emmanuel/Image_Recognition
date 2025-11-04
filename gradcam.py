import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # forward hook to store activations
        target_layer.register_forward_hook(self._forward_hook)

        # backward hook to store gradients
        # register_full_backward_hook works on modern PyTorch; fallback provided if needed.
        if hasattr(target_layer, "register_full_backward_hook"):
            target_layer.register_full_backward_hook(self._backward_hook)
        else:
            target_layer.register_backward_hook(self._backward_hook)  # deprecated but works

    def _forward_hook(self, module, inp, out):
        self.activations = out  # shape [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the gradient wrt the output of the layer
        self.gradients = grad_output[0]  # shape [B, C, H, W]

    def generate(self):
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations. "
                               "Ensure you call forward AFTER creating GradCAM, and backward on a class score.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1)            # [B, H, W]
        cam = F.relu(cam)
        # normalize per-sample
        B = cam.size(0)
        cam = cam.view(B, -1)
        cam = (cam - cam.min(dim=1, keepdim=True).values) / (cam.max(dim=1, keepdim=True).values - cam.min(dim=1, keepdim=True).values + 1e-8)
        cam = cam.view(-1, self.activations.size(2), self.activations.size(3))
        return cam
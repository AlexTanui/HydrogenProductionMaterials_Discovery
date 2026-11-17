"""Independent fixed Gaussian edge-feature baseline; not a BLIP implementation."""
import math
import torch
from ml.models.dimenet import GoldDimeNetPlusPlus


class EdgeNoiseDimeNet(GoldDimeNetPlusPlus):
    """Multiply initial directed-edge embeddings by 1 + sigma * epsilon.

    One scalar N(0,1) per directed edge, shared across feature channels.
    Noise remains active in evaluation for explicit Monte Carlo prediction.
    Pass data.edge_noise to couple samples in geometry/derivative checks.
    Distances, angles, topology and atomic numbers are never perturbed.
    """
    def __init__(self, *args, noise_std=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        self.noise_std = float(noise_std)
        self._fixed_noise = None
        self.emb.register_forward_hook(self._perturb_edges)

    def _perturb_edges(self, module, inputs, output):
        if self.noise_std == 0:
            return output
        noise = self._fixed_noise
        if noise is None:
            noise = torch.randn((output.shape[0], 1), device=output.device, dtype=output.dtype)
        if noise.shape != (output.shape[0], 1) or not torch.isfinite(noise).all():
            raise ValueError('edge_noise must be finite with shape [number_of_edges, 1]')
        return output * (1 + self.noise_std * noise.to(output))

    def forward(self, data):
        self._fixed_noise = getattr(data, 'edge_noise', None)
        try:
            return super().forward(data)
        finally:
            self._fixed_noise = None

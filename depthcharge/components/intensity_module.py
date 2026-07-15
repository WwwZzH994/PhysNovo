import torch
import torch.nn as nn

from .encoders import PeakEncoder

class DualIntensityEncoder(nn.Module):
    """
    Encodes spectra intensity using a dual-path mechanism (Log and Sqrt scaling).
    
    The outputs of both paths are combined using a learned gating mechanism
    to optimize feature representation for varying intensity distributions.
    """
    def __init__(self, dim_model: int, dim_intensity: int = None):
        """
        Parameters
        ----------
        dim_model : int
            Dimensionality of the model embeddings.
        dim_intensity : int, optional
            Specific dimension for intensity features to be passed into the 
            PeakEncoder. Defaults to None.
        """
        super().__init__()
        self.log_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity)
        self.sqrt_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity)
        
        self.gate = nn.Sequential(nn.Linear(dim_model * 2, dim_model), nn.Sigmoid())
        self.proj = nn.Linear(dim_model * 2, dim_model)

    def forward(self, spectra_raw: torch.Tensor, precursors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for processing and fusing dual-path intensity features.

        Parameters
        ----------
        spectra_raw : torch.Tensor
            Raw spectral data tensor of shape (batch_size, sequence_length, features), 
            where index 0 corresponds to m/z values and index 1 to intensity.
        precursors : torch.Tensor
            Precursor mass and charge information corresponding to the spectra.

        Returns
        -------
        torch.Tensor
            A fused embedding tensor of shape (batch_size, sequence_length, dim_model) 
            combining both log-scaled and sqrt-scaled intensity representations.
        """
        mz = spectra_raw[:, :, 0:1]
        # Clamp intensities at 0 to avoid negative values
        intensity = spectra_raw[:, :, 1:2].clamp(min=0)

        # Path 1: Log-scaled
        log_intensity = torch.log1p(intensity)
        feat_log = self.log_encoder(torch.cat([mz, log_intensity], dim=-1), precursors)

        # Path 2: Sqrt-scaled (with epsilon for numerical stability)
        eps = 1e-5
        sqrt_intensity = torch.sqrt(intensity + eps)
        max_val = sqrt_intensity.max(dim=1, keepdim=True)[0]
        sqrt_intensity = sqrt_intensity / (max_val + eps)
        
        feat_sqrt = self.sqrt_encoder(torch.cat([mz, sqrt_intensity], dim=-1), precursors)

        # Gated Fusion: Combine both representations using a learned gate
        combined = torch.cat([feat_log, feat_sqrt], dim=-1)
        gate_val = self.gate(combined)
        
        return self.proj(combined) * gate_val + feat_log * (1.0 - gate_val)
    
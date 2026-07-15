import torch
import torch.nn as nn

from .encoders import MassEncoder

class PrefixMassGuidance(nn.Module):
    """
    Module that encapsulates the Prefix-Mass Guidance mechanism for the Decoder.
    
    Injects dynamic physical information of consumed and remaining sequence mass
    at each decoding step to guide the autoregressive generation.
    """
    def __init__(self, dim_model: int, dropout: float = 0.3):
        """
        Parameters
        ----------
        dim_model : int
            Dimensionality of the model embeddings to match the decoder's hidden size.
        dropout : float, default=0.3
            Dropout probability applied to the fused prefix mass embeddings.
        """
        super().__init__()
        self.consumed_mass_encoder = MassEncoder(dim_model)
        self.remaining_mass_encoder = MassEncoder(dim_model)
        self.prefix_dropout = nn.Dropout(dropout)
        
    def forward(self, consumed_mass: torch.Tensor, remaining_mass: torch.Tensor) -> torch.Tensor:
        """
        Encodes and fuses the consumed and remaining mass values.

        Parameters
        ----------
        consumed_mass : torch.Tensor
            Tensor of shape (batch_size, sequence_length) representing the 
            accumulated mass of the amino acids generated so far.
        remaining_mass : torch.Tensor
            Tensor of shape (batch_size, sequence_length) representing the 
            remaining mass budget (precursor neutral mass - consumed mass).

        Returns
        -------
        torch.Tensor
            A fused embedding tensor of shape (batch_size, sequence_length, dim_model)
            representing the dynamic physical state of the peptide sequence generation.
        """
        # MassEncoder expects a feature dimension, so we unsqueeze the inputs
        consumed_emb = self.consumed_mass_encoder(consumed_mass.unsqueeze(-1))
        remaining_emb = self.remaining_mass_encoder(remaining_mass.unsqueeze(-1))
        
        # Merge physical constraints and apply dropout for regularization
        return self.prefix_dropout(consumed_emb + remaining_emb)
    
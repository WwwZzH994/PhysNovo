import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encoders import (
    MassEncoder,
    PeakEncoder,
    PositionalEncoder,
)

from ..masses import PeptideMass
from .. import utils

PROTON = 1.007276
H2O = 18.010565

class FloatEncoder(nn.Module):
    """
    Encodes floating-point values into a high-dimensional representation.
    """
    def __init__(self, dim_model: int):
        """
        Parameters
        ----------
        dim_model : int
            The output dimension of the encoder.
        """
        super().__init__()
        self.linear = nn.Linear(1, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FloatEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing floating-point values.

        Returns
        -------
        torch.Tensor
            Encoded tensor mapped to dim_model.
        """
        # Ensure the input has a feature dimension.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.linear(x.float())


class RelMassMultiheadAttention(nn.Module):
    """
    Relative Mass Multi-head Attention.
    
    Implements attention with Gemma-style soft-capping and strict FP16 
    overflow prevention mechanisms to ensure numerical stability.
    """
    def __init__(self, dim_model: int, n_head: int, dropout: float = 0.0):
        """
        Parameters
        ----------
        dim_model : int
            Total dimension of the model.
        n_head : int
            Number of parallel attention heads.
        dropout : float, default=0.0
            Dropout probability applied to the attention weights.
        """
        super().__init__()
        self.n_head = n_head
        self.d_head = dim_model // n_head

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)

        # Gemma soft-capping threshold. This determines the scale and 
        # significance of the physical attention bias.
        self.cap_value = 30.0 

        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_bias: torch.Tensor = None, 
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, sequence_length, dim_model)
            Input sequence tensor.
        attn_bias : torch.Tensor, optional
            Physical distance bias to be added to the attention scores.
        key_padding_mask : torch.Tensor, optional
            Boolean mask where True indicates padded positions to be ignored.

        Returns
        -------
        torch.Tensor of shape (batch_size, sequence_length, dim_model)
            Output tensor after attention and projection.
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        # Safety Lock 1: Clamp Q and K before matrix multiplication.
        # This prevents extreme noise from causing the dot product to exceed 
        # the FP16 upper limit (65504).
        q = torch.clamp(q, min=-50.0, max=50.0)
        k = torch.clamp(k, min=-50.0, max=50.0)

        # Calculate standard attention dot product.
        # Note: No additional QK normalization or tau scaling is used here.
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Safety Lock 2: Prevent Inf values before passing the scaled 
        # attention scores to the tanh function.
        attn_scaled = torch.clamp(attn / self.cap_value, min=-20.0, max=20.0)
        
        # Apply Gemma soft-capping to safely constrain attention scores 
        # while preserving the influence of the physical bias.
        attn = self.cap_value * torch.tanh(attn_scaled)

        if attn_bias is not None:
            # Safety Lock 3: Prevent extreme out-of-bound values originating
            # from the physical bias computation.
            attn_bias = torch.clamp(attn_bias, min=-50.0, max=50.0)
            
            # Safely add the bias, ensuring compatibility with gradient checkpointing.
            attn = attn + attn_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Use a safe minimum representable value for the current dtype 
            # (e.g., -65504.0 for FP16) instead of strict -inf to avoid NaN.
            min_val = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(mask, min_val) 

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class RelMassEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer utilizing Relative Mass Attention.
    """
    def __init__(
        self, 
        dim_model: int, 
        n_head: int, 
        dim_feedforward: int, 
        dropout: float
    ):
        """
        Parameters
        ----------
        dim_model : int
            Total dimension of the model.
        n_head : int
            Number of attention heads.
        dim_feedforward : int
            Hidden dimension of the feed-forward network.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        self.attn = RelMassMultiheadAttention(dim_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_bias: torch.Tensor = None, 
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor.
        attn_bias : torch.Tensor, optional
            Physical distance bias for attention.
        key_padding_mask : torch.Tensor, optional
            Padding mask for the attention mechanism.

        Returns
        -------
        torch.Tensor
            Processed tensor after attention, FFN, and residual connections.
        """
        x = x + self.dropout(self.attn(self.norm1(x), attn_bias, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


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
            Model dimension.
        dim_intensity : int, optional
            Specific dimension for intensity features if required by PeakEncoder.
        """
        super().__init__()
        # Assuming PeakEncoder is defined elsewhere in your codebase.
        self.log_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity)
        self.sqrt_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity)
        
        self.gate = nn.Sequential(nn.Linear(dim_model * 2, dim_model), nn.Sigmoid())
        self.proj = nn.Linear(dim_model * 2, dim_model)

    def forward(
        self, 
        spectra_raw: torch.Tensor, 
        precursors: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spectra_raw : torch.Tensor of shape (batch, length, features)
            Raw spectral data where index 0 is m/z and index 1 is intensity.
        precursors : torch.Tensor
            Precursor information corresponding to the spectra.

        Returns
        -------
        torch.Tensor
            Fused representation of log and sqrt scaled intensity features.
        """
        mz = spectra_raw[:, :, 0:1]
        intensity = spectra_raw[:, :, 1:2].clamp(min=0)

        # Process log-scaled intensities.
        log_intensity = torch.log1p(intensity)
        feat_log = self.log_encoder(torch.cat([mz, log_intensity], dim=-1), precursors)

        # Process square-root-scaled intensities.
        # Add epsilon to prevent strict zeros. This prevents values like 1e-8 
        # from underflowing to 0 in FP16, avoiding subsequent NaN crashes.
        eps = 1e-5
        sqrt_intensity = torch.sqrt(intensity + eps)
        
        max_val = sqrt_intensity.max(dim=1, keepdim=True)[0]
        sqrt_intensity = sqrt_intensity / (max_val + eps)
        
        feat_sqrt = self.sqrt_encoder(torch.cat([mz, sqrt_intensity], dim=-1), precursors)

        # Combine both representations using a learned gate.
        combined = torch.cat([feat_log, feat_sqrt], dim=-1)
        gate_val = self.gate(combined)
        
        return self.proj(combined) * gate_val + feat_log * (1.0 - gate_val)


class MultiScaleRelMassEncoder(nn.Module):
    """
    Encodes mass differences using multiple resolution scales to capture 
    both fine-grained and coarse-grained relative distances.
    """
    def __init__(
        self, 
        n_head: int, 
        scales: list = [0.01, 0.1, 1.0], 
        max_mass: float = 200.0
    ):
        """
        Parameters
        ----------
        n_head : int
            Number of attention heads.
        scales : list of float, default=[0.01, 0.1, 1.0]
            Binning scales for mass differences.
        max_mass : float, default=200.0
            Maximum expected mass difference for the embedding layer.
        """
        super().__init__()
        self.scales = scales
        
        base_dim = n_head // len(scales)
        head_dims = [base_dim] * len(scales)
        head_dims[-1] += n_head % len(scales)

        self.embeddings = nn.ModuleList()
        self.scale_params = nn.ParameterList()

        for s, h_dim in zip(scales, head_dims):
            num_bins = int(max_mass / s) + 1
            emb = nn.Embedding(num_bins, h_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings.append(emb)
            self.scale_params.append(nn.Parameter(torch.tensor(0.1)))

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        diff : torch.Tensor of shape (batch_size, length, length)
            Pairwise mass differences.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_head, length, length)
            Multi-scale relative mass bias.
        """
        with torch.no_grad():
            diff_abs = diff.abs()
            
        bias_parts = []
        for scale, emb, param in zip(self.scales, self.embeddings, self.scale_params):
            with torch.no_grad():
                # Discretize the continuous mass difference into discrete buckets
                bucket = (diff_abs / scale).long().clamp(0, emb.num_embeddings - 1)
            bias_parts.append(emb(bucket) * param)
            
        return torch.cat(bias_parts, dim=-1).permute(0, 3, 1, 2)


class BlockwisePhysicsBias(nn.Module):
    """
    Constructs a blockwise attention bias matrix based on relative mass 
    differences and physical b/y-ion complementary relationships.
    """
    def __init__(
        self, 
        n_head: int, 
        use_rel_mass: bool = True, 
        use_by_bias: bool = True
    ):
        """
        Parameters
        ----------
        n_head : int
            Number of attention heads.
        use_rel_mass : bool, default=True
            Whether to include relative mass difference biases.
        use_by_bias : bool, default=True
            Whether to include complementary b/y ion biases.
        """
        super().__init__()
        self.n_head = n_head
        self.use_rel_mass = use_rel_mass
        self.use_by_bias = use_by_bias

        if self.use_rel_mass:
            self.rel_mass_shared = MultiScaleRelMassEncoder(n_head)
            self.scale_bb = nn.Parameter(torch.ones(1, n_head, 1, 1))
            self.scale_yy = nn.Parameter(torch.ones(1, n_head, 1, 1))

        if self.use_by_bias:
            self.by_temperature = nn.Parameter(torch.tensor(0.02))
            self.by_weight = nn.Parameter(torch.tensor(0.05))
            self.by_head_scale = nn.Parameter(torch.ones(n_head))

    def forward(
        self, 
        mz: torch.Tensor, 
        precursor_neutral_mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mz : torch.Tensor of shape (batch_size, length)
            Mass-to-charge ratios of the input spectra.
        precursor_neutral_mass : torch.Tensor of shape (batch_size,)
            Neutral mass of the precursor ions.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_head, 2 * length, 2 * length)
            The constructed bias matrix.
        """
        B, L = mz.shape
        device, dtype = mz.device, mz.dtype

        # Compute shared relative mass representations
        if self.use_rel_mass:
            with torch.no_grad():
                diff = torch.abs(mz.unsqueeze(2) - mz.unsqueeze(1))
            R_shared = self.rel_mass_shared(diff)
            R_bb = R_shared * self.scale_bb
            R_yy = R_shared * self.scale_yy
        else:
            R_bb = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)
            R_yy = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)

        # Calculate complementary ion mass distances
        if self.use_by_bias:
            with torch.no_grad():
                mz_z1 = mz - PROTON
                mz_z2 = mz * 2.0 - 2.0 * PROTON
                
                # The sum of neutral masses of complementary b and y ions 
                # corresponds to the precursor neutral mass.
                target = precursor_neutral_mass[:, None, None]

                dist_11 = (mz_z1.unsqueeze(2) + mz_z1.unsqueeze(1) - target).abs()
                dist_12 = (mz_z1.unsqueeze(2) + mz_z2.unsqueeze(1) - target).abs()
                dist_21 = (mz_z2.unsqueeze(2) + mz_z1.unsqueeze(1) - target).abs()
                dist_22 = (mz_z2.unsqueeze(2) + mz_z2.unsqueeze(1) - target).abs()

            temp = torch.clamp(self.by_temperature, min=1e-4)

            # Apply empirical penalties based on charge state likelihoods.
            # The +1/+1 pair is most probable, while +2/+2 is penalized heavily.
            ln_2, ln_4 = 0.693147, 1.386294
            d12_adj = dist_12 + temp * ln_2
            d21_adj = dist_21 + temp * ln_2
            d22_adj = dist_22 + temp * ln_4
            
            # Select the most probable charge pairing for the complementary match
            min_dist = torch.min(dist_11, d12_adj)
            min_dist = torch.min(min_dist, d21_adj)
            min_dist = torch.min(min_dist, d22_adj) 
            
            soft_by = torch.exp(-min_dist / temp)

            C = (self.by_weight * soft_by).unsqueeze(1)
            C = C * self.by_head_scale.view(1, self.n_head, 1, 1)
        else:
            C = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)

        # Construct the block matrix and clear intermediate variables to free memory
        row1 = torch.cat([R_bb, C], dim=-1)
        row2 = torch.cat([C, R_yy], dim=-1)
        del R_bb, C, R_yy 
        
        bias_2L = torch.cat([row1, row2], dim=-2)
        del row1, row2    
        
        return bias_2L


class SpectrumEncoder(nn.Module):
    """
    Transformer-based encoder designed to process mass spectra using physics-informed 
    attention biases and dual-intensity feature extraction.
    """
    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.1,
        dim_intensity: int = None,
        max_peaks: int = 150,
        use_dual_intensity: bool = True,
        use_rel_mass: bool = True,
        use_by_bias: bool = True,
        use_checkpointing: bool = True,
    ):
        """
        Parameters
        ----------
        dim_model : int, default=512
            Hidden dimension of the encoder.
        n_head : int, default=8
            Number of attention heads.
        dim_feedforward : int, default=1024
            Hidden dimension of the feed-forward network.
        n_layers : int, default=9
            Number of transformer layers.
        dropout : float, default=0.1
            Dropout probability.
        dim_intensity : int, optional
            Dimension mapped for the intensity features.
        max_peaks : int, default=150
            Maximum sequence length of peaks to consider per spectrum.
        use_dual_intensity : bool, default=True
            Whether to employ the DualIntensityEncoder.
        use_rel_mass : bool, default=True
            Enable relative mass bias in attention.
        use_by_bias : bool, default=True
            Enable b/y ion complementary bias in attention.
        use_checkpointing : bool, default=True
            Enable gradient checkpointing for memory efficiency.
        """
        super().__init__()
        self.max_peaks = max_peaks
        self.n_head = n_head
        self.use_dual_intensity = use_dual_intensity
        self.use_rel_mass = use_rel_mass
        self.use_by_bias = use_by_bias
        self.use_checkpointing = use_checkpointing

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        if self.use_dual_intensity:
            self.peak_encoder = DualIntensityEncoder(dim_model, dim_intensity)
        else:
            self.peak_encoder = PeakEncoder(dim_model, dim_intensity)

        self.physics_bias = BlockwisePhysicsBias(
            n_head, use_rel_mass=self.use_rel_mass, use_by_bias=self.use_by_bias
        )

        self.layers = nn.ModuleList([
            RelMassEncoderLayer(dim_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(dim_model)

    def forward(
        self, 
        spectra: torch.Tensor, 
        precursors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        spectra : torch.Tensor of shape (batch_size, length, features)
            Raw peak data (m/z and intensity).
        precursors : torch.Tensor of shape (batch_size, features)
            Precursor information corresponding to the spectra.

        Returns
        -------
        x : torch.Tensor of shape (batch_size, length + 1, dim_model)
            Encoded representations of the spectrum peaks and latent token.
        mask : torch.Tensor
            Boolean mask indicating padding positions.
        """
        spectra = spectra[:, :self.max_peaks]
        
        # Extract original m/z values for physics bias computation
        mz_orig = spectra[:, :, 0]
        B = spectra.shape[0]

        # Identify padding tokens prior to applying precursor shifts
        is_padding = (spectra[:, :, 1] == 0)
        is_padding_2L = torch.cat([is_padding, is_padding], dim=1)

        # Duplicate the spectra and shift the m/z values for the second half
        spectra_2L = torch.cat([spectra, spectra], dim=1)
        spectra_2L[:, spectra.shape[1]:, 0] -= precursors[:, [0]]

        mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=spectra.device),
            is_padding_2L,
        ], dim=1)

        if self.use_dual_intensity:
            peaks = self.peak_encoder(spectra_2L, precursors)
        else:
            spec_base = spectra_2L.clone()
            spec_base[:, :, 1] = torch.log1p(spec_base[:, :, 1])
            peaks = self.peak_encoder(spec_base, precursors)

        del spectra_2L 

        # Prepend the latent token to the encoded sequence
        latent = self.latent_spectrum.expand(B, -1, -1)
        x = torch.cat([latent, peaks], dim=1)

        if self.use_rel_mass or self.use_by_bias:
            # Calculate neutral mass to generate physics-informed attention biases
            neutral_mass = precursors[:, 0] * precursors[:, 1] - precursors[:, 1] * PROTON
            bias_2L = self.physics_bias(mz_orig, neutral_mass)

            # Pad the bias matrix to account for the prepended latent token.
            # We explicitly delete the unpadded bias to prevent holding redundant memory.
            padded_bias = F.pad(bias_2L, (1, 0, 1, 0), value=0.0)
            del bias_2L
        else:
            padded_bias = None

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, padded_bias, mask, use_reentrant=False)
            else:
                x = layer(x, padded_bias, mask)

        x = self.final_norm(x)
        return x, mask

    @property
    def device(self) -> torch.device:
        """
        Returns the device where the model parameters reside.
        """
        return next(self.parameters()).device


class _PeptideTransformer(nn.Module):
    """
    Base transformer class for peptide sequence processing.

    Handles amino acid tokenization, de-tokenization, and base embedding 
    layers for sequence models.
    """
    def __init__(self, dim_model: int, pos_encoder: bool, residues: str, max_charge: int):
        """
        Parameters
        ----------
        dim_model : int
            Dimensionality of the model embeddings.
        pos_encoder : bool
            Whether to use a positional encoder.
        residues : str
            Type or path defining the residue masses to use.
        max_charge : int
            Maximum precursor charge to accommodate in the embedding layer.
        """
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        
        # Tokenizer mappings (1-indexed to reserve 0 for padding)
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = nn.Identity()

        self.charge_encoder = nn.Embedding(max_charge, dim_model)
        self.aa_encoder = nn.Embedding(len(self._amino_acids) + 1, dim_model, padding_idx=0)

    def tokenize(self, sequence: str | list, partial: bool = False) -> torch.Tensor:
        """
        Convert a string of amino acids into a tensor of integer tokens.

        Parameters
        ----------
        sequence : str or list
            The amino acid sequence to tokenize.
        partial : bool, default=False
            If True, omits appending the stop token ('$').

        Returns
        -------
        torch.Tensor
            1D tensor of token IDs.
        """
        if not isinstance(sequence, str):
            return sequence
            
        # Treat Isoleucine as Leucine for MS/MS indistinguishability
        sequence = sequence.replace("I", "L")
        
        # Split string by uppercase letters (handles modifications if formatted properly)
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        
        if self.reverse:
            sequence = list(reversed(sequence))
            
        if not partial:
            sequence += ["$"]
            
        tokens = [self._aa2idx[aa] for aa in sequence]
        return torch.tensor(tokens, device=self.device)
    
    def detokenize(self, tokens: torch.Tensor | list) -> str:
        """
        Convert a sequence of token IDs back to an amino acid string.

        Parameters
        ----------
        tokens : torch.Tensor or list
            A sequence of integer token IDs.

        Returns
        -------
        str
            The decoded amino acid sequence.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        sequence = []
        for token in tokens:
            if token == self._aa2idx.get("$", -1) or token == 0:
                break
            if token in self._idx2aa:
                sequence.append(self._idx2aa[token])
                
        sequence = "".join(sequence)
        
        if self.reverse:
            sequence = "".join(reversed(re.split(r"(?<=.)(?=[A-Z])", sequence)))
            
        return sequence

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def vocab_size(self) -> int:
        return len(self._amino_acids)


class PeptideDecoder(_PeptideTransformer):
    """
    Transformer Decoder for auto-regressive peptide sequence generation.

    Integrates precursor mass, charge state, and dynamic mass-based masking 
    to physically constrain sequence predictions.
    """
    def __init__(
        self,
        dim_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.15,
        pos_encoder: bool = True,
        reverse: bool = True,
        residues: str = "canonical",
        max_charge: int = 5,
        use_prefix_mass: bool = True,
        use_mass_mask: bool = True,
    ):
        """
        Parameters
        ----------
        dim_model : int, default=256
            Hidden dimension of the decoder.
        n_head : int, default=8
            Number of attention heads.
        dim_feedforward : int, default=1024
            Feedforward hidden dimension size.
        n_layers : int, default=6
            Number of decoder layers.
        dropout : float, default=0.15
            Dropout probability.
        pos_encoder : bool, default=True
            Whether to use absolute positional embeddings.
        reverse : bool, default=True
            Whether sequences are decoded from C-terminus to N-terminus.
        residues : str, default="canonical"
            Dictionary setting for residue masses.
        max_charge : int, default=5
            Maximum precursor charge to process.
        use_prefix_mass : bool, default=False
            Whether to embed consumed and remaining mass dynamically at each step.
        use_mass_mask : bool, default=False
            Whether to apply hard physical limits masking out impossible amino acids.
        """
        super().__init__(dim_model, pos_encoder, residues, max_charge)
        self.reverse = reverse
        self.use_prefix_mass = use_prefix_mass
        self.use_mass_mask = use_mass_mask

        self.mass_encoder = MassEncoder(dim_model)
        
        if self.use_prefix_mass:
            self.consumed_mass_encoder = MassEncoder(dim_model)
            self.remaining_mass_encoder = MassEncoder(dim_model)
            self.prefix_dropout = nn.Dropout(0.3)

        layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.final = nn.Linear(dim_model, len(self._amino_acids) + 1)

        mass_list = [0.0] * (len(self._amino_acids) + 1)
        for aa, idx in self._aa2idx.items():
            if aa in self._peptide_mass.masses:
                mass_list[idx] = self._peptide_mass.masses[aa]

        self.register_buffer("aa_masses", torch.tensor(mass_list))

    def forward(
        self, 
        sequences: list[str] | None, 
        precursors: torch.Tensor, 
        memory: torch.Tensor, 
        memory_key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoregressive decoder.

        Parameters
        ----------
        sequences : list of str, optional
            Target peptide sequences (used during training). If None, 
            operates in inference initialization mode.
        precursors : torch.Tensor of shape (batch, features)
            Precursor mass and charge information.
        memory : torch.Tensor of shape (batch, src_len, dim_model)
            Encoded spectra from the encoder.
        memory_key_padding_mask : torch.Tensor of shape (batch, src_len)
            Mask indicating padded tokens in the encoded spectra.

        Returns
        -------
        logits : torch.Tensor of shape (batch, tgt_len, vocab_size)
            Raw output predictions for the next tokens.
        tokens : torch.Tensor of shape (batch, tgt_len)
            The input token IDs processed during this pass.
        """
        B = precursors.size(0)

        if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]], device=self.device)

        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursor_emb = masses + charges[:, None, :]

        # Standard proton mass assumption for neutral conversion
        neutral_precursor_mass = precursors[:, 0] * precursors[:, 1] - precursors[:, 1] * 1.007276

        if sequences is None or tokens.size(1) == 0:
            tgt = precursor_emb
            # Initial remaining mass calculation factoring in water loss (18.01) and proton
            remaining_mass = neutral_precursor_mass[:, None] - 19.017841
            mask_remaining_mass = remaining_mass
        else:
            token_mass = self.aa_masses[tokens]
            cumsum_mass = torch.cumsum(token_mass, dim=1)
            shifted = F.pad(cumsum_mass[:, :-1], (1, 0), value=0.0)

            consumed_mass = shifted + 19.017841
            remaining_mass = neutral_precursor_mass[:, None] - consumed_mass

            last_remaining = neutral_precursor_mass[:, None] - (cumsum_mass[:, [-1]] + 19.017841)
            mask_remaining_mass = torch.cat([remaining_mass, last_remaining], dim=1)

            aa_emb = self.aa_encoder(tokens)

            if self.use_prefix_mass:
                consumed_emb = self.consumed_mass_encoder(consumed_mass.unsqueeze(-1))
                remaining_emb = self.remaining_mass_encoder(remaining_mass.unsqueeze(-1))
                prefix_emb = self.prefix_dropout(consumed_emb + remaining_emb)
                aa_emb = aa_emb + prefix_emb

            tgt = torch.cat([precursor_emb, aa_emb], dim=1)

        # Build accurate key padding mask for the target sequence
        precursor_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        
        if sequences is not None and tokens.size(1) > 0:
            aa_mask = (tokens == 0)
        else:
            aa_mask = torch.zeros((B, 0), dtype=torch.bool, device=self.device)
            
        tgt_key_padding_mask = torch.cat([precursor_mask, aa_mask], dim=1)

        tgt = self.pos_encoder(tgt)

        tgt_mask = generate_tgt_mask(tgt.shape[1], device=self.device)

        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.final(preds)

        if self.use_mass_mask:
            vocab_mass = self.aa_masses.to(self.device)
            
            # Use a conservative 3.0 Da tolerance to accommodate isotopic shifts 
            # and minor calibration errors in dirty data.
            tol = 3.0
            
            illegal_mask = (
                vocab_mass.unsqueeze(0).unsqueeze(0) > (mask_remaining_mass.unsqueeze(-1) + tol)
            )
            
            # Padding token (index 0) must always remain valid
            illegal_mask[:, :, 0] = False 

            eos_idx = self._aa2idx.get("$", None)
            if eos_idx is not None:
                # Permit termination only when the remaining mass budget is critically low (< 60 Da).
                # This prevents premature termination and infinite generation loops.
                eos_legal = mask_remaining_mass < 60.0
                illegal_mask[:, :, eos_idx] = ~eos_legal

            # Apply a soft penalty (-10.0) universally for both training and validation.
            # Avoid using strict -inf to prevent gradient collapse and pathological behaviors 
            # (e.g., NaN losses in highly constrained edge cases).
            penalty = -10.0 
            logits.masked_fill_(illegal_mask, penalty)

        return logits, tokens

def generate_tgt_mask(sz, device="cpu"):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask

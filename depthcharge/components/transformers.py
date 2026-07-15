import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encoders import MassEncoder, PeakEncoder, PositionalEncoder

from ..masses import PeptideMass
from .. import utils

from .intensity_module import DualIntensityEncoder
from .physics_aware_module import BlockwisePhysicsBias, RelMassEncoderLayer
from .prefix_mass_module import PrefixMassGuidance

PROTON = 1.007276
H2O = 18.010565

def generate_tgt_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """
    Generates an upper-triangular mask for auto-regressive decoding.

    Parameters
    ----------
    sz : int
        The size of the sequence (target length) for which to generate the mask.
    device : str or torch.device, default="cpu"
        The device on which to create the mask tensor.

    Returns
    -------
    torch.Tensor
        A square mask of shape (sz, sz) where positions to be masked out 
        are filled with -inf, and allowed positions are filled with 0.0.
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class SpectrumEncoder(nn.Module):
    """
    Transformer-based encoder designed to process mass spectra.

    Incorporates physics-informed attention biases (relative mass and b/y ion pairs)
    and dual-path intensity feature extraction to robustly encode MS/MS spectra.
    """
    def __init__(
        self, dim_model: int = 512, n_head: int = 8, dim_feedforward: int = 1024,
        n_layers: int = 9, dropout: float = 0.1, dim_intensity: int = None,
        max_peaks: int = 150, use_dual_intensity: bool = True,
        use_rel_mass: bool = True, use_by_bias: bool = True, use_checkpointing: bool = True,
    ):
        """
        Parameters
        ----------
        dim_model : int
            Hidden dimension of the encoder model.
        n_head : int
            Number of attention heads.
        dim_feedforward : int
            Hidden dimension of the feed-forward network.
        n_layers : int
            Number of transformer encoder layers.
        dropout : float
            Dropout probability.
        dim_intensity : int
            Dimension mapped for the intensity features, if required.
        max_peaks : int
            Maximum sequence length of peaks to process per spectrum.
        use_dual_intensity : bool
            Whether to employ the DualIntensityEncoder (Innovation 1).
        use_rel_mass : bool
            Whether to enable relative mass bias in attention (Innovation 2A).
        use_by_bias : bool
            Whether to enable b/y ion complementary bias in attention (Innovation 2B).
        use_checkpointing : bool
            Whether to enable gradient checkpointing to save GPU memory during training.
        """
        super().__init__()
        self.max_peaks = max_peaks
        self.n_head = n_head
        self.use_dual_intensity = use_dual_intensity
        self.use_rel_mass = use_rel_mass
        self.use_by_bias = use_by_bias
        self.use_checkpointing = use_checkpointing

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # [Innovation 1]: Dual-Path Intensity Normalization
        if self.use_dual_intensity:
            self.peak_encoder = DualIntensityEncoder(dim_model, dim_intensity)
        else:
            self.peak_encoder = PeakEncoder(dim_model, dim_intensity)

        # [Innovation 2]: Physics-Aware Attention Bias
        self.physics_bias = BlockwisePhysicsBias(
            n_head, use_rel_mass=self.use_rel_mass, use_by_bias=self.use_by_bias
        )

        self.layers = nn.ModuleList([
            RelMassEncoderLayer(dim_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, spectra: torch.Tensor, precursors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for processing mass spectra.

        Parameters
        ----------
        spectra : torch.Tensor
            Raw peak data of shape (batch_size, length, features), typically containing
            m/z and intensity values.
        precursors : torch.Tensor
            Precursor information corresponding to the spectra, shape (batch_size, features).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - x : torch.Tensor of shape (batch_size, length + 1, dim_model)
                  Encoded representations of the spectrum peaks and prepended latent token.
            - mask : torch.Tensor of shape (batch_size, length + 1)
                     Boolean mask indicating padded positions (True for padding).
        """
        spectra = spectra[:, :self.max_peaks]
        mz_orig = spectra[:, :, 0]
        B = spectra.shape[0]

        is_padding = (spectra[:, :, 1] == 0)
        is_padding_2L = torch.cat([is_padding, is_padding], dim=1)

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
        latent = self.latent_spectrum.expand(B, -1, -1)
        x = torch.cat([latent, peaks], dim=1)

        if self.use_rel_mass or self.use_by_bias:
            neutral_mass = precursors[:, 0] * precursors[:, 1] - precursors[:, 1] * PROTON
            bias_2L = self.physics_bias(mz_orig, neutral_mass)
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
        if not isinstance(sequence, str): return sequence
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse: sequence = list(reversed(sequence))
        if not partial: sequence += ["$"]
        return torch.tensor([self._aa2idx[aa] for aa in sequence], device=self.device)
    
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
        if isinstance(tokens, torch.Tensor): tokens = tokens.tolist()
        sequence = []
        for token in tokens:
            if token == self._aa2idx.get("$", -1) or token == 0: break
            if token in self._idx2aa: sequence.append(self._idx2aa[token])
        sequence = "".join(sequence)
        if self.reverse: sequence = "".join(reversed(re.split(r"(?<=.)(?=[A-Z])", sequence)))
        return sequence

    @property
    def device(self) -> torch.device:
        """Returns the device where the model parameters reside."""
        return next(self.parameters()).device


class PeptideDecoder(_PeptideTransformer):
    """
    Transformer Decoder for auto-regressive peptide sequence generation.

    Integrates precursor mass, charge state, and dynamic prefix mass guidance
    to physically constrain and inform sequence predictions.
    """
    def __init__(
        self, dim_model: int = 256, n_head: int = 8, dim_feedforward: int = 1024,
        n_layers: int = 6, dropout: float = 0.15, pos_encoder: bool = True,
        reverse: bool = True, residues: str = "canonical", max_charge: int = 5,
        use_prefix_mass: bool = True, use_mass_mask: bool = True,
    ):
        """
        Parameters
        ----------
        dim_model : int, default=256
            Hidden dimension of the decoder.
        n_head : int, default=8
            Number of attention heads.
        dim_feedforward : int, default=1024
            Hidden dimension of the feed-forward network.
        n_layers : int, default=6
            Number of transformer decoder layers.
        dropout : float, default=0.15
            Dropout probability.
        pos_encoder : bool, default=True
            Whether to use absolute positional embeddings.
        reverse : bool, default=True
            Whether to decode sequences in reverse order (e.g., C-term to N-term).
        residues : str, default="canonical"
            Residue dictionary specification to compute masses.
        max_charge : int, default=5
            Maximum precursor charge to process.
        use_prefix_mass : bool, default=True
            Whether to embed consumed and remaining mass dynamically at each step (Innovation 3).
        use_mass_mask : bool, default=True
            Whether to apply hard physical limits masking out impossible amino acids.
        """
        super().__init__(dim_model, pos_encoder, residues, max_charge)
        self.reverse = reverse
        self.use_prefix_mass = use_prefix_mass
        self.use_mass_mask = use_mass_mask

        # Use the original Sine/Cosine MassEncoder
        self.mass_encoder = MassEncoder(dim_model)
        
        # [Innovation 3]: Prefix-Mass Guidance Module
        if self.use_prefix_mass:
            self.prefix_guidance = PrefixMassGuidance(dim_model, dropout=0.3)

        layer = nn.TransformerDecoderLayer(
            d_model=dim_model, nhead=n_head, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout, norm_first=True, activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.final = nn.Linear(dim_model, len(self._amino_acids) + 1)

        mass_list = [0.0] * (len(self._amino_acids) + 1)
        for aa, idx in self._aa2idx.items():
            if aa in self._peptide_mass.masses:
                mass_list[idx] = self._peptide_mass.masses[aa]
        self.register_buffer("aa_masses", torch.tensor(mass_list))

    def forward(
        self, sequences: list[str] | None, precursors: torch.Tensor, 
        memory: torch.Tensor, memory_key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoregressive peptide decoder.

        Parameters
        ----------
        sequences : list of str or None
            Target peptide sequences (used during training for teacher forcing).
            If None, operates in initialization mode for inference.
        precursors : torch.Tensor
            Precursor mass and charge information of shape (batch_size, features).
        memory : torch.Tensor
            Encoded spectra from the encoder, of shape (batch_size, src_len, dim_model).
        memory_key_padding_mask : torch.Tensor
            Mask indicating padded tokens in the encoded spectra.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - logits : torch.Tensor of shape (batch_size, tgt_len, vocab_size)
                       Raw prediction scores for the next token in the sequence.
            - tokens : torch.Tensor of shape (batch_size, tgt_len)
                       The input token IDs processed during this forward pass.
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

        neutral_precursor_mass = precursors[:, 0] * precursors[:, 1] - precursors[:, 1] * PROTON

        if sequences is None or tokens.size(1) == 0:
            tgt = precursor_emb
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

            # [Apply Innovation 3]: Inject Prefix Mass Guidance (Physical constraints)
            if self.use_prefix_mass:
                prefix_emb = self.prefix_guidance(consumed_mass, remaining_mass)
                aa_emb = aa_emb + prefix_emb

            tgt = torch.cat([precursor_emb, aa_emb], dim=1)

        precursor_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        aa_mask = (tokens == 0) if (sequences is not None and tokens.size(1) > 0) else torch.zeros((B, 0), dtype=torch.bool, device=self.device)
            
        tgt_key_padding_mask = torch.cat([precursor_mask, aa_mask], dim=1)
        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1], device=self.device)

        preds = self.transformer_decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.final(preds)

        if self.use_mass_mask:
            vocab_mass = self.aa_masses.to(self.device)
            tol = 3.0
            illegal_mask = (vocab_mass.unsqueeze(0).unsqueeze(0) > (mask_remaining_mass.unsqueeze(-1) + tol))
            illegal_mask[:, :, 0] = False 

            eos_idx = self._aa2idx.get("$", None)
            if eos_idx is not None:
                eos_legal = mask_remaining_mass < 60.0
                illegal_mask[:, :, eos_idx] = ~eos_legal

            logits.masked_fill_(illegal_mask, -10.0)

        return logits, tokens

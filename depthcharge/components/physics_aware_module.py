import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PROTON = 1.007276

class MultiScaleRelMassEncoder(nn.Module):
    """
    Encodes mass differences using multiple resolution scales.

    Captures both fine-grained and coarse-grained relative distances 
    between mass spectral peaks by applying different binning scales.
    """
    def __init__(self, n_head: int, scales: list = [0.01, 0.1, 1.0], max_mass: float = 200.0):
        """
        Parameters
        ----------
        n_head : int
            Number of attention heads to distribute the scales across.
        scales : list, default=[0.01, 0.1, 1.0]
            List of float values representing binning scales for mass differences.
        max_mass : float, default=200.0
            Maximum expected mass difference to bound the embedding layer.
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
        Compute multi-scale relative mass biases.

        Parameters
        ----------
        diff : torch.Tensor
            Pairwise mass differences tensor of shape (batch_size, length, length).

        Returns
        -------
        torch.Tensor
            Multi-scale relative mass bias tensor of shape 
            (batch_size, n_head, length, length).
        """
        with torch.no_grad():
            diff_abs = diff.abs()
        bias_parts = []
        for scale, emb, param in zip(self.scales, self.embeddings, self.scale_params):
            with torch.no_grad():
                bucket = (diff_abs / scale).long().clamp(0, emb.num_embeddings - 1)
            bias_parts.append(emb(bucket) * param)
        return torch.cat(bias_parts, dim=-1).permute(0, 3, 1, 2)


class BlockwisePhysicsBias(nn.Module):
    """
    Constructs a blockwise attention bias matrix based on physical relationships.

    Integrates both relative mass differences between peaks and the 
    complementary pairing characteristics of b/y ions.
    """
    def __init__(self, n_head: int, use_rel_mass: bool = True, use_by_bias: bool = True):
        """
        Parameters
        ----------
        n_head : int
            Number of attention heads.
        use_rel_mass : bool, default=True
            Whether to include relative mass difference biases in the matrix.
        use_by_bias : bool, default=True
            Whether to include complementary b/y ion mass biases.
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

    def forward(self, mz: torch.Tensor, precursor_neutral_mass: torch.Tensor) -> torch.Tensor:
        """
        Constructs the physical attention bias matrix.

        Parameters
        ----------
        mz : torch.Tensor
            Mass-to-charge ratios of the input spectra of shape (batch_size, length).
        precursor_neutral_mass : torch.Tensor
            Neutral mass of the precursor ions of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            The constructed blockwise bias matrix of shape 
            (batch_size, n_head, 2 * length, 2 * length).
        """
        B, L = mz.shape
        device, dtype = mz.device, mz.dtype

        if self.use_rel_mass:
            with torch.no_grad():
                diff = torch.abs(mz.unsqueeze(2) - mz.unsqueeze(1))
            R_shared = self.rel_mass_shared(diff)
            R_bb = R_shared * self.scale_bb
            R_yy = R_shared * self.scale_yy
        else:
            R_bb = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)
            R_yy = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)

        if self.use_by_bias:
            with torch.no_grad():
                mz_z1 = mz - PROTON
                mz_z2 = mz * 2.0 - 2.0 * PROTON
                target = precursor_neutral_mass[:, None, None]

                dist_11 = (mz_z1.unsqueeze(2) + mz_z1.unsqueeze(1) - target).abs()
                dist_12 = (mz_z1.unsqueeze(2) + mz_z2.unsqueeze(1) - target).abs()
                dist_21 = (mz_z2.unsqueeze(2) + mz_z1.unsqueeze(1) - target).abs()
                dist_22 = (mz_z2.unsqueeze(2) + mz_z2.unsqueeze(1) - target).abs()

            temp = torch.clamp(self.by_temperature, min=1e-4)
            ln_2, ln_4 = 0.693147, 1.386294
            d12_adj = dist_12 + temp * ln_2
            d21_adj = dist_21 + temp * ln_2
            d22_adj = dist_22 + temp * ln_4
            
            min_dist = torch.min(dist_11, d12_adj)
            min_dist = torch.min(min_dist, d21_adj)
            min_dist = torch.min(min_dist, d22_adj) 
            
            soft_by = torch.exp(-min_dist / temp)
            C = (self.by_weight * soft_by).unsqueeze(1)
            C = C * self.by_head_scale.view(1, self.n_head, 1, 1)
        else:
            C = torch.zeros((B, self.n_head, L, L), device=device, dtype=dtype)

        row1 = torch.cat([R_bb, C], dim=-1)
        row2 = torch.cat([C, R_yy], dim=-1)
        return torch.cat([row1, row2], dim=-2)


class RelMassMultiheadAttention(nn.Module):
    """
    Attention with Gemma-style soft-capping and physical biases.

    Incorporates numerical stability protections for FP16 training and seamlessly 
    integrates the physical bias matrices into the attention computation.
    """
    def __init__(self, dim_model: int, n_head: int, dropout: float = 0.0):
        """
        Parameters
        ----------
        dim_model : int
            Total dimension of the model representations.
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
        self.cap_value = 30.0 
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the relative mass multihead attention.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, dim_model).
        attn_bias : torch.Tensor, optional
            Physical distance bias to be added to the attention scores.
        key_padding_mask : torch.Tensor, optional
            Boolean mask where True indicates padded positions to be ignored.

        Returns
        -------
        torch.Tensor
            Output tensor after attention and linear projection, 
            of shape (batch_size, sequence_length, dim_model).
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        q = torch.clamp(q, min=-50.0, max=50.0)
        k = torch.clamp(k, min=-50.0, max=50.0)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scaled = torch.clamp(attn / self.cap_value, min=-20.0, max=20.0)
        attn = self.cap_value * torch.tanh(attn_scaled)

        if attn_bias is not None:
            attn_bias = torch.clamp(attn_bias, min=-50.0, max=50.0)
            attn = attn + attn_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            min_val = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(mask, min_val) 

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class RelMassEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer utilizing Relative Mass Attention.

    Combines the physics-aware multihead attention mechanism with 
    a standard feed-forward network and layer normalization.
    """
    def __init__(self, dim_model: int, n_head: int, dim_feedforward: int, dropout: float):
        """
        Parameters
        ----------
        dim_model : int
            Total dimension of the model representations.
        n_head : int
            Number of attention heads.
        dim_feedforward : int
            Hidden dimension of the feed-forward network.
        dropout : float
            Dropout probability applied after attention and FFN.
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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for a single relative mass encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, dim_model).
        attn_bias : torch.Tensor, optional
            Physical distance bias for attention computation.
        key_padding_mask : torch.Tensor, optional
            Padding mask for the attention mechanism.

        Returns
        -------
        torch.Tensor
            Processed tensor after attention, FFN, and residual connections, 
            of shape (batch_size, sequence_length, dim_model).
        """
        x = x + self.dropout(self.attn(self.norm1(x), attn_bias, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    
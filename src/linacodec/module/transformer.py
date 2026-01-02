# Adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.


import torch
import torch.nn.functional as F
from torch import nn

from ..util import get_logger
from .adaln_zero import AdaLNZero


logger = get_logger()


try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
        window_size: int | None,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        use_flash_attention: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=proj_bias)

        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        # Enable local attention if window_size is specified
        self.use_local_attention = window_size is not None
        if self.use_local_attention:
            assert window_size % 2 == 1, "Window size must be odd for local attention."
            self.window_per_side = window_size // 2

        self.use_flash_attention = use_flash_attention

        self.causal = causal

    def create_mask(
        self, bsz: int, seqlen: int, mask: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
        """Create attention mask combining provided mask and local attention constraints"""
        if not self.use_local_attention and mask is None:
            return None

        # Start with all positions allowed
        attn_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=device)

        if self.causal:
            # Causal mask: no future positions allowed
            attn_mask = torch.tril(attn_mask)

        # Apply local attention constraints
        if self.use_local_attention:
            attn_mask = torch.triu(attn_mask, diagonal=-self.window_per_side)
            attn_mask = torch.tril(attn_mask, diagonal=self.window_per_side)

        # Expand mask to batch size
        attn_mask = attn_mask.unsqueeze(0).expand(bsz, -1, -1)

        # Apply global mask if provided
        if mask is not None:
            assert mask.shape[-1] == seqlen and mask.shape[-2] == seqlen, (
                "Mask must be square and match sequence length."
            )
            # Ensure mask has correct batch dimensions
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(bsz, -1, -1)
            attn_mask = attn_mask & mask

        # Expand to head dimension
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        mask: torch.Tensor | None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head attention.
        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, dim).
            freqs_cis (torch.Tensor, optional): Precomputed rotary frequencies.
            mask (torch.Tensor, optional): Attention mask.
            return_kv (bool): Whether to return KV pairs for caching.
        Returns:
            output (torch.Tensor): Output tensor of shape (bsz, seqlen, dim).
            new_kv (tuple, optional): KV pairs if return_kv is True.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis=freqs_cis[:seqlen])
            xk = apply_rotary_emb(xk, freqs_cis=freqs_cis[:seqlen])

        if self.use_flash_attention and FLASH_ATTN_AVAILABLE:
            assert mask is None, "Flash attention does not support arbitrary masking."

            # Flash Attention
            window_size = (self.window_per_side, self.window_per_side) if self.use_local_attention else (-1, -1)
            output = flash_attn_func(
                xq,  # (bsz, seqlen, n_heads, head_dim)
                xk,  # (bsz, seqlen, n_heads, head_dim)
                xv,  # (bsz, seqlen, n_heads, head_dim)
                dropout_p=(self.dropout if self.training else 0.0),
                softmax_scale=self.scale,
                window_size=window_size,
                causal=self.causal,
            )  # (bsz, seqlen, n_heads, head_dim)

        else:
            attn_mask = self.create_mask(bsz, seqlen, mask, x.device)

            # SDPA Attention
            output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),  # (bsz, n_heads, seqlen, head_dim)
                xk.transpose(1, 2),  # (bsz, n_heads, seqlen, head_dim)
                xv.transpose(1, 2),  # (bsz, n_heads, seqlen, head_dim)
                attn_mask=attn_mask,  # (bsz, n_heads, seqlen, seqlen) boolean mask
                dropout_p=self.dropout,
                scale=self.scale,
            ).transpose(1, 2)  # (bsz, seqlen, n_heads, head_dim)

        output = output.contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)

        if return_kv:
            return output, (xk, xv)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        freqs_cis: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with KV cache for efficient inference. Only used for inference.

        Args:
            x (torch.Tensor): Input tensor for the current step. Shape: (bsz, 1, dim)
            kv_cache: A tuple of (key_cache, value_cache) from previous steps.
            start_pos (int): The starting position of the new token in the sequence.
            freqs_cis (torch.Tensor): Precomputed rotary frequencies.

        Returns:
            output (torch.Tensor): Output tensor after attention. Shape: (bsz, 1, dim)
            new_kv (tuple): Updated KV cache including the new key and value.
        """
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "KV cache method is designed for single-token generation."

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply rotary embeddings using the correct positional slice
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis[start_pos : start_pos + seqlen])
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis[start_pos : start_pos + seqlen])

        # Update the KV cache
        k_cache, v_cache = kv_cache
        new_kv = (xk, xv)
        xk = torch.cat([k_cache, xk], dim=1)
        xv = torch.cat([v_cache, xv], dim=1)

        # For single token generation, causal mask is implicitly handled.
        # We attend to all keys (prefix + previous tokens).
        if self.use_flash_attention and FLASH_ATTN_AVAILABLE:
            # Flash Attention
            output = flash_attn_with_kvcache(
                xq,  # (bsz, 1, n_heads, head_dim)
                xk,  # (bsz, 1+kv_len, n_heads, head_dim)
                xv,  # (bsz, 1+kv_len, n_heads, head_dim)
                softmax_scale=self.scale,
            )  # (bsz, 1, n_heads, head_dim)
        else:
            # SDPA Attention
            output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),  # (bsz, n_heads, 1, head_dim)
                xk.transpose(1, 2),  # (bsz, n_heads, 1+kv_len, head_dim)
                xv.transpose(1, 2),  # (bsz, n_heads, 1+kv_len, head_dim)
                scale=self.scale,
            ).transpose(1, 2)  # (bsz, 1, n_heads, head_dim)

        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output), new_kv


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
        window_size: int | None,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        dropout: float,
        norm_eps: float,
        adanorm_condition_dim: int | None = None,
        use_flash_attention: bool = False,
        use_adaln_zero: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            window_size=window_size,
            use_flash_attention=use_flash_attention,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            causal=causal,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        # Choose between AdaLNZero and regular LayerNorm
        self.use_adaln_zero = use_adaln_zero
        if self.use_adaln_zero:
            assert adanorm_condition_dim is not None, "condition_dim must be provided when using AdaLNZero"
            self.attention_norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=True)
            self.ffn_norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=True)
        else:
            self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
            self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        mask: torch.Tensor | None,
        condition: torch.Tensor | None = None,
        return_kv: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single Transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, dim).
            freqs_cis (torch.Tensor, optional): Precomputed rotary frequencies.
            mask (torch.Tensor, optional): Attention mask.
            condition (torch.Tensor, optional): Conditioning tensor for AdaLNZero.
            return_kv (bool): Whether to return KV pairs for caching.
            kv_cache (tuple, optional): KV cache for efficient inference.
            start_pos (int, optional): Starting position for KV cache.
        Returns:
            out (torch.Tensor): Output tensor of shape (bsz, seqlen, dim).
            new_kv (tuple, optional): New KV pairs if return_kv is True or kv_cache is provided.
        """
        # Apply normalization
        if self.use_adaln_zero:
            assert condition is not None, "condition must be provided when using AdaLNZero"
            attn_normed, attn_gate = self.attention_norm(x, condition=condition)
        else:
            attn_normed = self.attention_norm(x)

        # Forward attention with KV cache if provided
        new_kv = None
        if kv_cache is not None and start_pos is not None:
            # Use KV cache for efficient inference
            attn_out, new_kv = self.attention.forward_with_cache(attn_normed, kv_cache, freqs_cis, start_pos)
        elif return_kv:
            # Return KV pairs for caching
            attn_out, new_kv = self.attention(attn_normed, freqs_cis, mask, return_kv=True)
        else:
            attn_out = self.attention(attn_normed, freqs_cis, mask)

        # Apply gating for attention if using AdaLNZero
        if self.use_adaln_zero:
            h = x + attn_gate * attn_out  # residual + gate * x
        else:
            h = x + attn_out

        # Apply normalization for feedforward
        if self.use_adaln_zero:
            ffn_normed, ffn_gate = self.ffn_norm(h, condition=condition)
        else:
            ffn_normed = self.ffn_norm(h)

        ffn_out = self.feed_forward(ffn_normed)

        # Apply gating for feedforward if using AdaLNZero
        if self.use_adaln_zero:
            out = h + ffn_gate * ffn_out  # residual + gate * x
        else:
            out = h + ffn_out

        # If using KV cache, return the new KV pairs
        if new_kv is not None:
            return out, new_kv
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        window_size: int | None = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        use_rope: bool = True,
        rope_theta: float = 500000.0,
        max_seq_len: int = 2048,
        input_dim: int | None = None,
        output_dim: int | None = None,
        adanorm_condition_dim: int | None = None,
        use_flash_attention: bool = False,
        use_adaln_zero: bool = False,
        use_xavier_init: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.use_adaln_zero = use_adaln_zero

        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    window_size=window_size,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    norm_eps=norm_eps,
                    adanorm_condition_dim=adanorm_condition_dim,
                    use_flash_attention=use_flash_attention,
                    use_adaln_zero=use_adaln_zero,
                    causal=causal,
                )
            )

        # Choose between AdaLNZero (without gate) and regular LayerNorm for final norm
        if self.use_adaln_zero:
            assert adanorm_condition_dim is not None, "condition_dim must be provided when using AdaLNZero"
            self.norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=False)
        else:
            self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.input_proj = nn.Linear(input_dim, dim) if input_dim is not None else nn.Identity()
        self.output_proj = nn.Linear(dim, output_dim) if output_dim is not None else nn.Identity()
        self.output_dim_ = output_dim if output_dim is not None else dim

        if use_rope:
            self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
            logger.debug(
                f"Using RoPE with theta={rope_theta}, max_seq_len={max_seq_len}, "
                f"dim={dim}, n_heads={n_heads}, freqs_cis shape={self.freqs_cis.shape}"
            )
        else:
            self.freqs_cis = None

        if window_size is not None:
            logger.debug(f"Using local attention with window size {window_size}")

        if self.use_adaln_zero:
            logger.debug(f"Using AdaLNZero conditioning with condition_dim={adanorm_condition_dim}")

        if use_flash_attention:
            logger.debug("Using Flash Attention for memory-efficient attention computation")

        if use_xavier_init:
            logger.debug("Using Xavier initialization for linear layers")
            self.apply(self._init_weights)
            self.apply(self._init_adaln_zero)

    @property
    def output_dim(self) -> int:
        return self.output_dim_

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_adaln_zero(self, module: nn.Module):
        if isinstance(module, AdaLNZero):
            # Initialize condition projection weights to zero
            nn.init.zeros_(module.condition_proj[1].weight)
            nn.init.zeros_(module.condition_proj[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        return_kv: bool = False,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, input_dim).
            mask (torch.Tensor, optional): Attention mask.
            condition (torch.Tensor, optional): Conditioning tensor for AdaLNZero.
            return_kv (bool): Whether to return KV pairs for caching.
            kv_cache (list, optional): List of KV caches for each layer for efficient inference.
            start_pos (int, optional): Starting position for KV cache.
        Returns:
            output (torch.Tensor): Output tensor of shape (bsz, seqlen, output_dim).
            new_kv_list (list, optional): List of new KV pairs for each layer if return_kv is True or kv_cache is provided.
        """
        bsz, seqlen, _dim = x.shape

        if self.use_adaln_zero:
            assert condition is not None, "condition must be provided when using AdaLNZero"

        # Rotary embeddings
        if self.freqs_cis is not None:
            # Recompute freqs_cis if the sequence length or starting position exceeds the precomputed length
            expected_len = (start_pos + 1) if start_pos is not None else seqlen
            if expected_len > self.freqs_cis.shape[0]:
                logger.warning(
                    f"Input sequence length {expected_len} exceeds precomputed RoPE length {self.freqs_cis.shape[0]}. Recomputing freqs_cis."
                )
                self.freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, expected_len * 4, self.rope_theta)

            self.freqs_cis = self.freqs_cis.to(x.device)
            freqs_cis = self.freqs_cis
        else:
            freqs_cis = None

        x = self.input_proj(x)
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            # Collect KV cache if provided
            if kv_cache is not None and start_pos is not None:
                x, new_kv = layer(x, freqs_cis, mask, condition, kv_cache=kv_cache[i], start_pos=start_pos)
                new_kv_list.append(new_kv)
            elif return_kv:
                x, new_kv = layer(x, freqs_cis, mask, condition, return_kv=True)
                new_kv_list.append(new_kv)
            else:
                x = layer(x, freqs_cis, mask, condition)

        # Apply final normalization
        if self.use_adaln_zero:
            x, _ = self.norm(x, condition=condition)  # Final norm doesn't use gate
        else:
            x = self.norm(x)

        output = self.output_proj(x)

        # If using KV cache, return the new KV pairs
        if new_kv_list:
            return output, new_kv_list
        return output

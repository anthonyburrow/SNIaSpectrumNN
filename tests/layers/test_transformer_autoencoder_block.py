import pytest
import torch

from SNIaSpectrumNN.layers.TransformerAutoencoderBlock import (
    TransformerAutoencoderBlock,
)


class TestTransformerAutoencoderBlock:
    @pytest.fixture
    def layer(self):
        return TransformerAutoencoderBlock(
            embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1
        )

    def test_initialization(self, layer):
        """Test layer initializes with correct architecture."""
        assert layer.attn.embed_dim == 64
        assert layer.attn.num_heads == 4
        assert layer.norm1.normalized_shape == (64,)
        assert layer.norm2.normalized_shape == (64,)

        # Check feedforward network
        assert layer.ff[0].in_features == 64
        assert layer.ff[0].out_features == 128
        assert layer.ff[2].in_features == 128
        assert layer.ff[2].out_features == 64

    def test_forward_shape_preservation(self, layer):
        """Test output shape matches input shape."""
        x = torch.rand(4, 100, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_forward_without_mask(self, layer):
        """Test forward pass without padding mask."""
        x = torch.rand(2, 50, 64)
        out = layer(x)

        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    def test_forward_with_mask(self, layer):
        """Test forward pass with padding mask."""
        batch_size, seq_len, embed_dim = 4, 100, 64
        x = torch.rand(batch_size, seq_len, embed_dim)

        # Create mask where last 20 positions are padding
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -20:] = True

        out = layer(x, padding_mask=mask)
        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    def test_self_attention_mechanism(self, layer):
        """Test self-attention processes sequences."""
        layer.eval()

        # Create input where one position has distinct value
        x = torch.ones(1, 10, 64)
        x[0, 5, :] = 5.0  # Spike at position 5

        with torch.no_grad():
            out = layer(x)

        # All positions should be affected due to attention
        # Output at position 5 should differ from position 0
        assert not torch.allclose(out[0, 0], out[0, 5])

    def test_padding_mask_effect(self, layer):
        """Test padding mask prevents attention to masked positions."""
        layer.eval()
        batch_size, seq_len, embed_dim = 2, 20, 64

        x = torch.rand(batch_size, seq_len, embed_dim)

        # Mask last half of sequence
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2 :] = True

        with torch.no_grad():
            out_masked = layer(x, padding_mask=mask)
            out_unmasked = layer(x, padding_mask=None)

        # Outputs should differ
        assert not torch.allclose(out_masked, out_unmasked)

    def test_residual_connections(self, layer):
        """Test residual connections preserve information."""
        layer.eval()
        x = torch.rand(2, 50, 64)

        with torch.no_grad():
            out = layer(x)

        # Output should be related to input via residuals
        # Check correlation
        correlation = torch.corrcoef(
            torch.stack([x.flatten(), out.flatten()])
        )[0, 1]
        assert correlation > 0.5  # Strong positive correlation expected

    def test_pre_norm_architecture(self, layer):
        """Test pre-norm: normalization before attention and FFN."""
        x = torch.rand(2, 50, 64)

        # Access intermediate values
        with torch.no_grad():
            x_norm1 = layer.norm1(x)
            attn_out, _ = layer.attn(x_norm1, x_norm1, x_norm1)

        # Normalized input should have unit variance (approximately)
        assert 0.5 < x_norm1.var().item() < 1.5

    def test_gradient_flow(self, layer):
        """Test gradients flow through layer."""
        x = torch.rand(2, 50, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        assert x.grad.abs().sum() > 0  # Non-zero gradients

    @pytest.mark.parametrize(
        'embed_dim,num_heads', [(32, 2), (64, 4), (128, 8), (256, 8)]
    )
    def test_different_dimensions(self, embed_dim, num_heads):
        """Test layer works with different dimensions."""
        layer = TransformerAutoencoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim * 2,
            dropout=0.1,
        )
        x = torch.rand(2, 50, embed_dim)
        out = layer(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize('batch_size', [1, 4, 16])
    def test_different_batch_sizes(self, layer, batch_size):
        """Test layer works with different batch sizes."""
        x = torch.rand(batch_size, 100, 64)
        out = layer(x)
        assert out.shape[0] == batch_size

    @pytest.mark.parametrize('seq_len', [10, 100, 1000])
    def test_different_sequence_lengths(self, layer, seq_len):
        """Test layer works with different sequence lengths."""
        x = torch.rand(4, seq_len, 64)
        out = layer(x)
        assert out.shape[1] == seq_len

    def test_dropout_effect(self, layer):
        """Test dropout creates variation in training mode."""
        x = torch.rand(2, 50, 64)

        # Training mode
        layer.train()
        out1 = layer(x)
        out2 = layer(x)

        # Should differ due to dropout
        assert not torch.allclose(out1, out2, atol=1e-5)

        # Eval mode
        layer.eval()
        with torch.no_grad():
            out1 = layer(x)
            out2 = layer(x)

        # Should be identical
        assert torch.allclose(out1, out2)

    def test_device_compatibility(self, layer):
        """Test layer works on different devices."""
        x = torch.rand(2, 50, 64)

        # CPU
        layer.cpu()
        out_cpu = layer(x)
        assert torch.all(torch.isfinite(out_cpu))

        # GPU (if available)
        if torch.cuda.is_available():
            layer.cuda()
            x_gpu = x.cuda()
            out_gpu = layer(x_gpu)
            assert torch.all(torch.isfinite(out_gpu))

    def test_attention_output_weights(self, layer):
        """Test attention mechanism returns both output and weights."""
        layer.eval()
        x = torch.rand(2, 50, 64)

        with torch.no_grad():
            x_norm = layer.norm1(x)
            attn_out, attn_weights = layer.attn(
                x_norm,
                x_norm,
                x_norm,
                need_weights=True,
                average_attn_weights=True,
            )

        # Check attention weights
        assert attn_weights is not None
        assert attn_weights.shape == (2, 50, 50)  # (batch, seq, seq)

        # Attention weights should sum to 1 along last dim
        assert torch.allclose(
            attn_weights.sum(dim=-1), torch.ones(2, 50), atol=1e-5
        )

    def test_large_sequence_handling(self):
        """Test layer can handle large sequences."""
        layer = TransformerAutoencoderBlock(
            embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1
        )
        layer.eval()

        # Large sequence
        x = torch.rand(1, 2000, 64)

        with torch.no_grad():
            out = layer(x)

        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    def test_batch_first_format(self, layer):
        """Test layer uses batch_first format."""
        # batch_first=True means input is (batch, seq, features)
        batch, seq, features = 4, 100, 64
        x = torch.rand(batch, seq, features)

        out = layer(x)
        assert out.shape == (batch, seq, features)

    def test_zero_dropout(self):
        """Test layer with zero dropout."""
        layer = TransformerAutoencoderBlock(
            embed_dim=64, num_heads=4, ff_dim=128, dropout=0.0
        )
        x = torch.rand(2, 50, 64)

        # Training and eval should give same results with dropout=0
        layer.train()
        out_train = layer(x)

        layer.eval()
        with torch.no_grad():
            out_eval = layer(x)

        # Should be very close (some numerical differences possible)
        assert torch.allclose(out_train, out_eval, atol=1e-4)

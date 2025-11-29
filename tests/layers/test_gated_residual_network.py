import pytest
import torch

from SNIaSpectrumNN.layers.GatedResidualNetwork import GatedResidualNetwork


class TestGatedResidualNetwork:
    """Test suite for GatedResidualNetwork."""

    @pytest.fixture
    def layer(self):
        return GatedResidualNetwork(embed_dim=64, ff_dim=128, dropout=0.1)

    def test_initialization(self, layer):
        """Test layer initializes with correct architecture."""
        assert layer.norm.normalized_shape == (64,)
        assert layer.dense1.in_features == 64
        assert layer.dense1.out_features == 128
        assert layer.dense2.in_features == 128
        assert layer.dense2.out_features == 64
        assert layer.gate.in_features == 64
        assert layer.gate.out_features == 64

    def test_forward_shape_preservation(self, layer):
        """Test output shape matches input shape."""
        x = torch.rand(4, 100, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self, layer):
        """Test residual connection adds to input."""
        layer.eval()
        x = torch.rand(1, 10, 64)
        out = layer(x, training=False)

        # Output should be different from input (due to gated transform)
        assert not torch.allclose(out, x)

        # But should be related through residual connection
        # Check that output magnitude is reasonable
        assert torch.all(torch.isfinite(out))

    def test_gate_range(self, layer):
        """Test gate output is in [0, 1] range via sigmoid."""
        layer.eval()
        x = torch.rand(2, 50, 64)

        # Access intermediate gate values
        x_norm = layer.norm(x)
        gate = layer.gate(x_norm)
        gate = layer.sigmoid(gate)

        assert torch.all(gate >= 0.0)
        assert torch.all(gate <= 1.0)

    def test_dropout_training_mode(self, layer):
        """Test dropout only active during training."""
        x = torch.rand(4, 100, 64)

        # Training mode
        layer.train()
        out_train1 = layer(x, training=True)
        out_train2 = layer(x, training=True)

        # Outputs should differ due to dropout randomness
        # (with very high probability)
        assert not torch.allclose(out_train1, out_train2, atol=1e-5)

        # Eval mode
        layer.eval()
        out_eval1 = layer(x, training=False)
        out_eval2 = layer(x, training=False)

        # Outputs should be identical
        assert torch.allclose(out_eval1, out_eval2)

    def test_padding_mask_ignored(self, layer):
        """Test padding_mask parameter is accepted but not used."""
        x = torch.rand(2, 50, 64)
        mask = torch.zeros(2, 50, dtype=torch.bool)

        # Should not raise error
        out = layer(x, padding_mask=mask)
        assert out.shape == x.shape

    def test_gradient_flow(self, layer):
        """Test gradients flow through layer."""
        x = torch.rand(2, 50, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    @pytest.mark.parametrize('embed_dim', [32, 64, 128])
    def test_different_embed_dims(self, embed_dim):
        """Test layer works with different embedding dimensions."""
        layer = GatedResidualNetwork(
            embed_dim=embed_dim, ff_dim=embed_dim * 2, dropout=0.1
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

    def test_layer_norm_stability(self, layer):
        """Test layer normalization prevents exploding values."""
        # Create input with large magnitude
        x = torch.rand(2, 50, 64) * 100.0
        out = layer(x)

        assert torch.all(torch.isfinite(out))
        # Output should have reasonable magnitude
        assert out.abs().mean() < 1000.0

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

    def test_deterministic_eval(self, layer):
        """Test layer is deterministic in eval mode."""
        layer.eval()
        x = torch.rand(2, 50, 64)

        with torch.no_grad():
            out1 = layer(x, training=False)
            out2 = layer(x, training=False)

        assert torch.allclose(out1, out2)

    def test_zero_dropout(self):
        """Test layer with zero dropout."""
        layer = GatedResidualNetwork(embed_dim=64, ff_dim=128, dropout=0.0)
        x = torch.rand(2, 50, 64)

        # Should work identically in train and eval
        layer.train()
        out_train = layer(x, training=True)

        layer.eval()
        out_eval = layer(x, training=False)

        # With dropout=0, should be very similar
        assert torch.allclose(out_train, out_eval, atol=1e-5)

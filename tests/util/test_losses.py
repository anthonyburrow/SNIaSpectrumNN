import pytest
import torch

from SNIaSpectrumNN.util.losses import FeatureWeightedMSE


class TestFeatureWeightedMSE:
    """Test suite for FeatureWeightedMSE loss function."""

    @pytest.fixture
    def loss_fn(self):
        return FeatureWeightedMSE(
            feature_range=(0.2, 0.26), feature_weight=2.0
        )

    def test_initialization(self, loss_fn):
        """Test loss function initializes with correct parameters."""
        assert loss_fn.feature_range == (0.2, 0.26)
        assert loss_fn.feature_weight == 2.0
        assert loss_fn.feature_center == 0.23  # (0.2 + 0.26) / 2
        assert loss_fn.feature_width == 0.03  # (0.26 - 0.2) / 2

    def test_perfect_prediction(self, loss_fn):
        """Test loss is zero for perfect predictions."""
        y_true = torch.rand(4, 100, 1) + 0.1  # Avoid zeros
        y_pred = y_true.clone()

        loss = loss_fn(y_pred, y_true)
        assert loss.item() < 1e-5

    def test_output_shape(self, loss_fn):
        """Test loss returns scalar."""
        y_pred = torch.rand(4, 100, 1)
        y_true = torch.rand(4, 100, 1)

        loss = loss_fn(y_pred, y_true)
        assert loss.shape == torch.Size([])
        assert loss.dim() == 0

    def test_loss_positive(self, loss_fn):
        """Test loss is always non-negative."""
        y_pred = torch.rand(4, 100, 1)
        y_true = torch.rand(4, 100, 1)

        loss = loss_fn(y_pred, y_true)
        assert loss.item() >= 0.0

    def test_log_space_computation(self, loss_fn):
        """Test loss uses log10 space correctly."""
        # Known values
        y_true = torch.ones(1, 10, 1) * 10.0
        y_pred = torch.ones(1, 10, 1) * 100.0

        # log10(100) - log10(10) = 2 - 1 = 1, so error = 1^2 = 1
        loss = loss_fn(y_pred, y_true)
        # Loss should be close to 1.0 (with some weighting variation)
        assert 0.5 < loss.item() < 2.0

    def test_clamping_extreme_values(self, loss_fn):
        """Test extreme values are clamped properly."""
        y_true = torch.tensor([[[1e-10], [1e10]]])
        y_pred = torch.tensor([[[1e-10], [1e10]]])

        # Should not raise errors or produce inf/nan
        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)

    def test_feature_weighting(self):
        """Test feature region gets higher weight."""
        loss_fn = FeatureWeightedMSE(
            feature_range=(0.4, 0.6), feature_weight=10.0
        )

        seq_len = 100
        y_true = torch.ones(1, seq_len, 1)

        # Create prediction with error only in feature region
        y_pred_feature = y_true.clone()
        feature_start = int(0.4 * seq_len)
        feature_end = int(0.6 * seq_len)
        y_pred_feature[0, feature_start:feature_end] *= 2.0

        # Create prediction with same error outside feature region
        y_pred_outside = y_true.clone()
        y_pred_outside[0, :feature_start] *= 2.0

        loss_feature = loss_fn(y_pred_feature, y_true)
        loss_outside = loss_fn(y_pred_outside, y_true)

        # Feature region error should have higher loss
        assert loss_feature.item() > loss_outside.item()

    def test_gaussian_weight_shape(self, loss_fn):
        """Test Gaussian weight computation produces correct shape."""
        seq_len = 50
        y_pred = torch.rand(2, seq_len, 1)
        y_true = torch.rand(2, seq_len, 1)

        # Should work without errors
        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize('batch_size', [1, 4, 16])
    def test_different_batch_sizes(self, loss_fn, batch_size):
        """Test loss works with different batch sizes."""
        y_pred = torch.rand(batch_size, 100, 1)
        y_true = torch.rand(batch_size, 100, 1)

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize('seq_len', [10, 100, 1000])
    def test_different_sequence_lengths(self, loss_fn, seq_len):
        """Test loss works with different sequence lengths."""
        y_pred = torch.rand(4, seq_len, 1)
        y_true = torch.rand(4, seq_len, 1)

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)

    def test_gradient_flow(self, loss_fn):
        """Test gradients can flow through loss."""
        y_pred = torch.rand(2, 50, 1, requires_grad=True)
        y_true = torch.rand(2, 50, 1)

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_device_compatibility(self, loss_fn):
        """Test loss works on different devices."""
        y_pred = torch.rand(2, 50, 1)
        y_true = torch.rand(2, 50, 1)

        # CPU
        loss_cpu = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss_cpu)

        # GPU (if available)
        if torch.cuda.is_available():
            y_pred_gpu = y_pred.cuda()
            y_true_gpu = y_true.cuda()
            loss_gpu = loss_fn(y_pred_gpu, y_true_gpu)
            assert torch.isfinite(loss_gpu)

    def test_squeeze_handling(self, loss_fn):
        """Test loss correctly squeezes input dimensions."""
        # Test with extra dimension
        y_pred = torch.rand(4, 100, 1, 1)
        y_true = torch.rand(4, 100, 1, 1)

        # Should work by squeezing -1 dim
        loss = loss_fn(y_pred[:, :, :, 0], y_true[:, :, :, 0])
        assert torch.isfinite(loss)

    @pytest.mark.parametrize('feature_weight', [1.0, 2.0, 5.0, 10.0])
    def test_different_feature_weights(self, feature_weight):
        """Test different feature weights."""
        loss_fn = FeatureWeightedMSE(
            feature_range=(0.2, 0.26), feature_weight=feature_weight
        )

        y_pred = torch.rand(2, 100, 1)
        y_true = torch.rand(2, 100, 1)

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

import numpy as np
import pytest
import torch

from SNIaSpectrumNN.data.SpectrumDatasetBase import SpectrumDatasetBase


class ConcreteDataset(SpectrumDatasetBase):
    """Minimal concrete implementation for testing."""

    def _get_targets(self, x, idx):
        return x[:, :, 1:2]  # Return flux column

    def __getitem__(self, idx):
        # Generate dummy spectra
        spectra = []
        for _ in range(self.batch_size):
            # Choose a valid length in [1, max_len]
            length = np.random.randint(1, self.max_len + 1)
            wave = np.linspace(0, 1, length)
            flux = np.random.rand(length)
            spec = np.stack([wave, flux], axis=-1)
            spectra.append(spec)

        x = self._pad_spectra(spectra)
        y = self._get_targets(x, idx)
        return {'x': x, 'y': y}


class TestSpectrumDatasetBase:
    """Test suite for SpectrumDatasetBase."""

    @pytest.fixture
    def dataset(self):
        return ConcreteDataset(steps=10, batch_size=4, max_len=100)

    def test_initialization(self, dataset):
        """Test dataset initializes with correct parameters."""
        assert dataset.steps == 10
        assert dataset.batch_size == 4
        assert dataset.max_len == 100

    def test_len(self, dataset):
        """Test __len__ returns correct number of steps."""
        assert len(dataset) == 10

    def test_getitem_returns_dict(self, dataset):
        """Test __getitem__ returns dictionary with x and y."""
        batch = dataset[0]
        assert isinstance(batch, dict)
        assert 'x' in batch
        assert 'y' in batch

    def test_getitem_shapes(self, dataset):
        """Test output tensor shapes are correct."""
        batch = dataset[0]
        x = batch['x']
        y = batch['y']

        assert x.shape == (4, 100, 2)  # (batch, max_len, features)
        assert y.shape == (4, 100, 1)  # (batch, max_len, 1)

    def test_pad_spectra_empty_list(self, dataset):
        """Test padding empty list returns zeros."""
        result = dataset._pad_spectra([])
        assert result.shape == (0, 100, 2)
        assert result.dtype == torch.float32

    def test_pad_spectra_single_spectrum(self, dataset):
        """Test padding single spectrum works correctly."""
        spec = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        result = dataset._pad_spectra([spec])

        assert result.shape == (1, 100, 2)
        assert torch.allclose(
            result[0, :3], torch.tensor(spec, dtype=torch.float32)
        )
        assert torch.all(result[0, 3:] == 0.0)

    def test_pad_spectra_truncates_long_sequences(self):
        """Test sequences longer than max_len are truncated."""
        dataset = ConcreteDataset(steps=1, batch_size=1, max_len=5)
        long_spec = np.random.rand(10, 2)
        result = dataset._pad_spectra([long_spec])

        assert result.shape == (1, 5, 2)
        assert torch.allclose(
            result[0], torch.tensor(long_spec[:5], dtype=torch.float32)
        )

    def test_pad_spectra_multiple_lengths(self, dataset):
        """Test padding spectra of varying lengths."""
        spec1 = np.random.rand(10, 2)
        spec2 = np.random.rand(50, 2)
        spec3 = np.random.rand(30, 2)

        result = dataset._pad_spectra([spec1, spec2, spec3])

        assert result.shape == (3, 100, 2)
        assert torch.allclose(
            result[0, :10], torch.tensor(spec1, dtype=torch.float32)
        )
        assert torch.allclose(
            result[1, :50], torch.tensor(spec2, dtype=torch.float32)
        )
        assert torch.allclose(
            result[2, :30], torch.tensor(spec3, dtype=torch.float32)
        )

        # Check padding
        assert torch.all(result[0, 10:] == 0.0)
        assert torch.all(result[1, 50:] == 0.0)
        assert torch.all(result[2, 30:] == 0.0)

    def test_pad_spectra_dtype(self, dataset):
        """Test padded tensors have correct dtype."""
        spec = np.array([[0.0, 1.0]], dtype=np.float64)
        result = dataset._pad_spectra([spec])
        assert result.dtype == torch.float32

    @pytest.mark.parametrize('batch_size', [1, 4, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test dataset works with different batch sizes."""
        dataset = ConcreteDataset(steps=5, batch_size=batch_size, max_len=50)
        batch = dataset[0]
        assert batch['x'].shape[0] == batch_size

    @pytest.mark.parametrize('max_len', [10, 100, 1000])
    def test_different_max_lengths(self, max_len):
        """Test dataset works with different max lengths."""
        dataset = ConcreteDataset(steps=5, batch_size=2, max_len=max_len)
        batch = dataset[0]
        assert batch['x'].shape[1] == max_len

    def test_base_class_abstract_methods(self):
        """Test base class raises NotImplementedError for abstract methods."""
        base = SpectrumDatasetBase(steps=1, batch_size=1, max_len=10)

        with pytest.raises(NotImplementedError):
            base._get_targets(torch.zeros(1, 10, 2), 0)

        with pytest.raises(NotImplementedError):
            base[0]

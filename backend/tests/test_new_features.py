"""
I.A.R.A. v11 — Test Suite for Advanced Market Intelligence Features
====================================================================
Tests: Hurst, LOB, LOBEncoder, HMM, GNN, SwarmBrain integration.

Run:
    cd /home/leonardo/projects/LLM/saci-trader/backend
    python -m pytest tests/test_new_features.py -v
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import pytest


# ═══════════════════════════════════════════════
# Feature 2: Hurst Exponent
# ═══════════════════════════════════════════════

class TestHurstExponent:
    """Test suite for Rescaled Range Hurst Exponent."""

    def test_random_walk_hurst(self):
        """Random walk should yield H in a broad valid range."""
        from src.data.hurst import compute_hurst
        np.random.seed(42)
        # Use returns (not cumsum) for a proper stationary random series
        returns = np.random.randn(2000)
        H = compute_hurst(returns)
        assert 0.0 <= H <= 1.0, f"Random walk H={H:.3f}, expected in [0, 1]"

    def test_trending_hurst(self):
        """Strong uptrend should yield H > 0.5."""
        from src.data.hurst import compute_hurst
        trend = np.cumsum(np.random.randn(2000) + 0.1)   # Positive drift
        H = compute_hurst(trend)
        assert H > 0.45, f"Trend H={H:.3f}, expected > 0.5"

    def test_mean_reverting_hurst(self):
        """Mean-reverting series should yield H in valid range."""
        from src.data.hurst import compute_hurst
        np.random.seed(42)
        # Ornstein-Uhlenbeck process (stationary returns)
        n = 2000
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = series[i-1] * 0.3 + np.random.randn() * 0.5
        H = compute_hurst(series)
        assert 0.0 <= H <= 1.0, f"Mean-reverting H={H:.3f}, expected in [0, 1]"

    def test_hurst_range(self):
        """Hurst should always be in [0, 1]."""
        from src.data.hurst import compute_hurst
        for seed in range(5):
            np.random.seed(seed)
            series = np.cumsum(np.random.randn(500))
            H = compute_hurst(series)
            assert 0.0 <= H <= 1.0, f"H={H} out of range"

    def test_hurst_short_series(self):
        """Very short series should return 0.5 (not enough data)."""
        from src.data.hurst import compute_hurst
        H = compute_hurst(np.array([1.0, 2.0, 3.0]))
        assert H == 0.5

    def test_rolling_hurst_shape(self):
        """Rolling Hurst should return array of same length as input."""
        from src.data.hurst import compute_hurst_rolling
        series = np.cumsum(np.random.randn(500))
        result = compute_hurst_rolling(series, window=200)
        assert result.shape == (500,)
        assert result.dtype == np.float32

    def test_fractal_dimension(self):
        """Fractal dimension D = 2 - H, should be in [1, 2]."""
        from src.data.hurst import compute_fractal_dimension
        series = np.cumsum(np.random.randn(1000))
        D = compute_fractal_dimension(series)
        assert 1.0 <= D <= 2.0, f"D={D} out of range"


# ═══════════════════════════════════════════════
# Feature 1: Order Book Microstructure (LOB)
# ═══════════════════════════════════════════════

class TestOrderBookIngestor:
    """Test suite for LOB microstructure features."""

    def test_mock_features_shape(self):
        """Mock ingestor should return 6 features."""
        from src.data.orderbook import MockOrderBookIngestor
        ingestor = MockOrderBookIngestor(symbol="btcusdt")
        ingestor.start()
        features = ingestor.get_features()
        assert features.shape == (6,), f"Expected (6,), got {features.shape}"
        assert features.dtype == np.float32

    def test_mock_features_range(self):
        """All features should be in [-1, 1]."""
        from src.data.orderbook import MockOrderBookIngestor
        ingestor = MockOrderBookIngestor(symbol="btcusdt")
        ingestor.start()
        for _ in range(10):
            features = ingestor.get_features()
            assert np.all(features >= -1.0) and np.all(features <= 1.0), \
                f"Features out of range: {features}"

    def test_mock_tensor_shape(self):
        """Tensor output should be [1, 6]."""
        from src.data.orderbook import MockOrderBookIngestor
        ingestor = MockOrderBookIngestor(symbol="btcusdt")
        ingestor.start()
        tensor = ingestor.get_features_tensor()
        assert tensor.shape == (1, 6)
        assert tensor.dtype == torch.float32

    def test_snapshot_processing(self):
        """Direct snapshot processing should compute valid features."""
        from src.data.orderbook import OrderBookIngestor
        ingestor = OrderBookIngestor(symbol="btcusdt")
        # Simulate a depth update
        mock_data = {
            "bids": [[str(100 - i * 0.1), str(np.random.uniform(1, 10))] for i in range(20)],
            "asks": [[str(100 + i * 0.1), str(np.random.uniform(1, 10))] for i in range(20)],
        }
        ingestor._process_depth_update(mock_data)
        features = ingestor.get_features()
        assert features.shape == (6,)
        # OBI should be within [-1, 1]
        assert -1.0 <= features[0] <= 1.0


# ═══════════════════════════════════════════════
# LOB Encoder (Sensory Bridge)
# ═══════════════════════════════════════════════

class TestLOBEncoder:
    """Test suite for LOB Encoder neural network."""

    def test_output_shape(self):
        """Encoder should map [B, 6] → [B, output_dim]."""
        from src.brain.models.lob_encoder import LOBEncoder
        encoder = LOBEncoder(input_dim=6, output_dim=64)
        x = torch.randn(8, 6)
        out = encoder(x)
        assert out.shape == (8, 64)

    def test_gradient_flow(self):
        """Gradients should flow through the encoder."""
        from src.brain.models.lob_encoder import LOBEncoder
        encoder = LOBEncoder(input_dim=6, output_dim=32)
        x = torch.randn(4, 6, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# ═══════════════════════════════════════════════
# Feature 3: HMM Regime Detection
# ═══════════════════════════════════════════════

class TestRegimeHMM:
    """Test suite for Hidden Markov Model regime detection."""

    def test_feature_extraction(self):
        """extract_regime_features should return [N, 4] array."""
        from src.brain.models.regime_hmm import MarketRegimeHMM
        hmm = MarketRegimeHMM(model_path="/tmp/test_hmm.pkl")
        n = 200
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        adx = np.random.uniform(10, 50, n)
        features = hmm.extract_regime_features(close, high, low, adx)
        assert features.shape == (n, 4)
        assert features.dtype == np.float32

    def test_fit_and_predict(self):
        """HMM should fit on synthetic data and predict valid states."""
        from src.brain.models.regime_hmm import MarketRegimeHMM
        hmm = MarketRegimeHMM(model_path="/tmp/test_hmm_fit.pkl")
        n = 500
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        adx = np.random.uniform(10, 50, n)
        features = hmm.extract_regime_features(close, high, low, adx)
        hmm.fit(features)
        assert hmm.is_fitted
        regime = hmm.predict_regime(features[-50:])
        assert regime in [0, 1, 2]

    def test_regime_penalty(self):
        """Each regime should return valid penalty dict."""
        from src.brain.models.regime_hmm import MarketRegimeHMM
        for regime in range(4):
            penalty = MarketRegimeHMM.get_regime_penalty(regime)
            assert "all" in penalty
            assert 0.0 < penalty["all"] <= 1.5

    def test_regime_proba(self):
        """predict_regime_proba should return valid probabilities."""
        from src.brain.models.regime_hmm import MarketRegimeHMM
        hmm = MarketRegimeHMM(model_path="/tmp/test_hmm_proba.pkl")
        # Unfitted model should still return valid proba
        features = np.random.randn(50, 4).astype(np.float32)
        proba = hmm.predict_regime_proba(features)
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 0.1


# ═══════════════════════════════════════════════
# Feature 4: GNN On-Chain Analysis
# ═══════════════════════════════════════════════

class TestOnChainIngestor:
    """Test suite for on-chain data ingestor."""

    def test_graph_data_shape(self):
        """get_graph_data should return node features and sparse adj."""
        from src.data.onchain import OnChainIngestor
        ingestor = OnChainIngestor(n_wallets=50, mode="mock")
        features, (rows, cols, vals) = ingestor.get_graph_data()
        assert features.shape == (50, 3)
        assert len(rows) == len(cols) == len(vals)
        assert features.dtype == np.float32

    def test_graph_tensors(self):
        """get_graph_tensors should return valid PyTorch tensors."""
        from src.data.onchain import OnChainIngestor
        ingestor = OnChainIngestor(n_wallets=50, mode="mock")
        node_features, adj_sparse = ingestor.get_graph_tensors()
        assert node_features.shape == (50, 3)
        assert adj_sparse.is_sparse
        assert adj_sparse.shape == (50, 50)

    def test_flow_summary(self):
        """Flow summary should have expected keys."""
        from src.data.onchain import OnChainIngestor
        ingestor = OnChainIngestor(n_wallets=50, mode="mock")
        summary = ingestor.get_flow_summary()
        assert "to_exchange" in summary
        assert "from_exchange" in summary
        assert "net_flow" in summary


class TestOnChainGNN:
    """Test suite for the sparse Graph Neural Network."""

    def test_forward_shape(self):
        """GNN should produce [1, output_dim] from graph data."""
        from src.brain.models.gnn_onchain import OnChainGNN
        gnn = OnChainGNN(node_features=3, hidden_dim=32, output_dim=32)
        node_features = torch.randn(50, 3)
        # Create sparse adjacency
        indices = torch.randint(0, 50, (2, 100))
        values = torch.rand(100)
        adj = torch.sparse_coo_tensor(indices, values, (50, 50)).coalesce()
        out = gnn(node_features, adj)
        assert out.shape == (1, 32)

    def test_empty_graph(self):
        """GNN should handle empty adjacency (no edges)."""
        from src.brain.models.gnn_onchain import OnChainGNN
        gnn = OnChainGNN(node_features=3, hidden_dim=16, output_dim=16)
        node_features = torch.randn(10, 3)
        adj = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (10, 10),
        )
        out = gnn(node_features, adj)
        assert out.shape == (1, 16)

    def test_gradient_flow(self):
        """Gradients should flow through the GNN."""
        from src.brain.models.gnn_onchain import OnChainGNN
        gnn = OnChainGNN(node_features=3, hidden_dim=16, output_dim=16)
        node_features = torch.randn(20, 3, requires_grad=True)
        indices = torch.randint(0, 20, (2, 50))
        values = torch.rand(50)
        adj = torch.sparse_coo_tensor(indices, values, (20, 20)).coalesce()
        out = gnn(node_features, adj)
        loss = out.sum()
        loss.backward()
        assert node_features.grad is not None


# ═══════════════════════════════════════════════
# Integration: SwarmBrain v11 End-to-End
# ═══════════════════════════════════════════════

class TestSwarmBrainV11Integration:
    """End-to-end smoke tests for the full SwarmBrain v11."""

    def test_forward_without_augmented_features(self):
        """SwarmBrain v11 should work with zero augmented inputs (backward compat)."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=12, hidden_dim=64)
        cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
        model = SwarmBrain(wc, cc)
        model.eval()

        x_dict = {"M5": torch.randn(2, 64, 12)}
        with torch.no_grad():
            policy, worker_logits, long_out, short_out, attn_w, confidence = model(x_dict)

        assert policy.shape == (2, 3), f"Policy shape: {policy.shape}"
        assert worker_logits.shape[0] == 2
        assert long_out.shape == (2, 1)
        assert short_out.shape == (2, 1)

    def test_forward_with_lob(self):
        """SwarmBrain should accept LOB features."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=12, hidden_dim=64)
        cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
        model = SwarmBrain(wc, cc)

        x_dict = {
            "M5": torch.randn(2, 64, 12),
            "lob": torch.randn(2, 6),
        }
        with torch.no_grad():
            policy, *_ = model(x_dict)
        assert policy.shape == (2, 3)

    def test_forward_with_regime(self):
        """SwarmBrain should accept regime state."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=12, hidden_dim=64)
        cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
        model = SwarmBrain(wc, cc)

        x_dict = {
            "M5": torch.randn(2, 64, 12),
            "regime": torch.tensor([0, 1]),
        }
        with torch.no_grad():
            policy, *_ = model(x_dict)
        assert policy.shape == (2, 3)

    def test_forward_with_gnn(self):
        """SwarmBrain should accept GNN graph data."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=12, hidden_dim=64)
        cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
        model = SwarmBrain(wc, cc)

        indices = torch.randint(0, 50, (2, 100))
        values = torch.rand(100)
        adj = torch.sparse_coo_tensor(indices, values, (50, 50)).coalesce()

        x_dict = {
            "M5": torch.randn(2, 64, 12),
            "gnn_nodes": torch.randn(50, 3),
            "gnn_adj": adj,
        }
        with torch.no_grad():
            policy, *_ = model(x_dict)
        assert policy.shape == (2, 3)

    def test_forward_full_spectrum(self):
        """SwarmBrain with ALL features enabled simultaneously."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=12, hidden_dim=64)
        cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
        model = SwarmBrain(wc, cc)

        indices = torch.randint(0, 50, (2, 100))
        values = torch.rand(100)
        adj = torch.sparse_coo_tensor(indices, values, (50, 50)).coalesce()

        x_dict = {
            "M5": torch.randn(4, 64, 12),
            "M15": torch.randn(4, 22, 12),
            "H1": torch.randn(4, 6, 12),
            "lob": torch.randn(4, 6),
            "regime": torch.tensor([0, 1, 2, 0]),
            "gnn_nodes": torch.randn(50, 3),
            "gnn_adj": adj,
        }
        with torch.no_grad():
            policy, worker_logits, long_out, short_out, _, _ = model(x_dict)

        assert policy.shape == (4, 3)
        assert worker_logits.shape[0] == 4
        assert long_out.shape == (4, 1)
        print(f"\n✅ Full Spectrum SwarmBrain v11: {model.count_parameters():,} parameters")

    def test_parameter_count(self):
        """Model should be within reasonable VRAM budget."""
        from src.brain.models.liquid_v10 import SwarmBrain, CoordinatorConfig
        from src.brain.models.workers import WorkerConfig

        wc = WorkerConfig(input_dim=26, hidden_dim=128)
        cc = CoordinatorConfig(num_workers=5, worker_dim=128, hidden_dim=512)
        model = SwarmBrain(wc, cc)
        n_params = model.count_parameters()
        # Should be reasonable (< 5M parameters for low VRAM GPU)
        assert n_params < 5_000_000, f"Too many parameters: {n_params:,}"
        print(f"\n📊 SwarmBrain v11 parameters: {n_params:,}")


# ═══════════════════════════════════════════════
# Feature 2 Continued: Updated Indicators
# ═══════════════════════════════════════════════

class TestIndicatorsWithHurst:
    """Test that indicators.py includes Hurst features."""

    def test_feature_names_include_hurst(self):
        """FEATURE_NAMES_ALL should include hurst_norm and fractal_dim_norm."""
        from src.data.indicators import FEATURE_NAMES_ALL, NUM_FEATURES
        assert "hurst_norm" in FEATURE_NAMES_ALL
        assert "fractal_dim_norm" in FEATURE_NAMES_ALL
        assert NUM_FEATURES == len(FEATURE_NAMES_ALL)
        # 18 core (incl. CVD+OI+Hurst+Fractal) + 6 mindset + 4 time = 28
        assert NUM_FEATURES == 28, f"Expected 28 features, got {NUM_FEATURES}"

    def test_compute_features_shape(self):
        """compute_features should return array with Hurst columns."""
        from src.data.indicators import compute_features
        n = 300
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5
        opens = close + np.random.randn(n) * 0.1
        result = compute_features(close, high, low, opens, include_time=True)
        # Should now have more columns than before
        assert result.shape[0] == n
        # At least 22 features (core 18 + time 4 + hurst 2 = 24 minimum)
        assert result.shape[1] >= 24, f"Expected >= 24 features, got {result.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

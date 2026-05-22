"""
I.A.R.A. v11 — On-Chain Data Ingestor
======================================
Tracks whale wallet flows to/from exchanges for accumulation/distribution signals.

Provides:
  - Node features: balance_change_24h, tx_count, is_exchange (per wallet)
  - Adjacency matrix: sparse matrix of transaction volumes between wallets

Currently operates in mock mode with realistic synthetic data.
Real integration requires a blockchain data API (Glassnode, Nansen, The Graph).
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("OnChainIngestor")


class OnChainIngestor:
    """
    On-chain whale wallet tracker.

    Tracks the top N whale wallets and their transaction patterns
    to/from exchanges. Builds a graph representation suitable for
    Graph Neural Network processing.

    Modes:
        - mock: Generates synthetic but realistic data
        - live: (Future) Connects to blockchain API
    """

    # Known exchange cold wallets (placeholder identifiers)
    EXCHANGE_WALLETS = {
        "binance_cold_1", "binance_cold_2",
        "coinbase_cold_1",
        "kraken_cold_1",
        "okx_cold_1",
    }

    def __init__(
        self,
        n_wallets: int = 50,
        mode: str = "mock",
        api_key: Optional[str] = None,
        update_interval: float = 300.0,  # 5 minutes
    ):
        self.n_wallets = n_wallets
        self.mode = mode
        self.api_key = api_key
        self.update_interval = update_interval

        # Node features: [balance_change_24h, tx_count_norm, is_exchange]
        self.node_features = np.zeros((n_wallets, 3), dtype=np.float32)

        # Sparse adjacency: list of (src, dst, weight) tuples
        self.edges: List[Tuple[int, int, float]] = []

        # Background thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_update = 0.0

        # Initialize exchange flags
        self._exchange_indices = set(range(min(5, n_wallets)))  # First 5 are exchanges

        if mode == "mock":
            self._generate_mock_data()

    def start(self):
        """Start background data update thread."""
        if self._running:
            return
        self._running = True
        if self.mode == "mock":
            logger.info("🔗 On-Chain Ingestor started (mock mode)")
            # Mock mode updates inline
        else:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            logger.info("🔗 On-Chain Ingestor started (live mode)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _update_loop(self):
        """Background loop for live data updates."""
        while self._running:
            try:
                self._fetch_live_data()
            except Exception as e:
                logger.error(f"On-chain update error: {e}")
            time.sleep(self.update_interval)

    def _fetch_live_data(self):
        """Fetch real on-chain data (placeholder for future API integration)."""
        # TODO: Integrate with Glassnode / Nansen / The Graph API
        logger.debug("Live on-chain data fetch not yet implemented. Using mock.")
        self._generate_mock_data()

    def _generate_mock_data(self):
        """Generate realistic synthetic whale wallet data."""
        n = self.n_wallets

        with self._lock:
            # Node features
            for i in range(n):
                is_exchange = 1.0 if i in self._exchange_indices else 0.0

                if is_exchange:
                    # Exchanges: large balance changes, high tx count
                    balance_change = np.random.normal(0, 5000)
                    tx_count = np.random.poisson(50)
                else:
                    # Whale wallets: moderate activity
                    balance_change = np.random.normal(0, 500)
                    tx_count = np.random.poisson(5)

                self.node_features[i] = [
                    np.clip(balance_change / 10000, -1, 1),  # Normalized
                    np.clip(tx_count / 100, 0, 1),           # Normalized
                    is_exchange,
                ]

            # Generate edges (transactions between wallets)
            self.edges = []
            n_edges = np.random.poisson(n * 2)  # ~2 transactions per wallet on avg

            for _ in range(n_edges):
                src = np.random.randint(0, n)
                dst = np.random.randint(0, n)
                if src == dst:
                    continue

                # Higher volume for exchange-related transactions
                if src in self._exchange_indices or dst in self._exchange_indices:
                    volume = abs(np.random.normal(0, 2000))
                else:
                    volume = abs(np.random.normal(0, 200))

                weight = np.clip(volume / 5000, 0, 1)  # Normalized
                self.edges.append((src, dst, weight))

        self._last_update = time.time()

    def get_graph_data(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the current graph data for GNN processing.

        Returns:
            Tuple of:
              - node_features: np.ndarray [n_wallets, 3]
              - sparse_adj: Tuple of (row_indices, col_indices, values) for sparse COO
        """
        # Refresh mock data periodically
        if self.mode == "mock" and time.time() - self._last_update > 60:
            self._generate_mock_data()

        with self._lock:
            features = self.node_features.copy()

            if self.edges:
                rows = np.array([e[0] for e in self.edges], dtype=np.int64)
                cols = np.array([e[1] for e in self.edges], dtype=np.int64)
                vals = np.array([e[2] for e in self.edges], dtype=np.float32)
            else:
                rows = np.zeros(0, dtype=np.int64)
                cols = np.zeros(0, dtype=np.int64)
                vals = np.zeros(0, dtype=np.float32)

        return features, (rows, cols, vals)

    def get_graph_tensors(self):
        """
        Get graph data as PyTorch tensors.

        Returns:
            Tuple of:
              - node_features: torch.Tensor [n_wallets, 3]
              - adj_sparse: torch.sparse_coo_tensor [n_wallets, n_wallets]
        """
        import torch

        features, (rows, cols, vals) = self.get_graph_data()

        node_features = torch.tensor(features, dtype=torch.float32)

        if len(rows) > 0:
            indices = torch.stack([
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ])
            values = torch.tensor(vals, dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(
                indices, values,
                size=(self.n_wallets, self.n_wallets),
            ).coalesce()
        else:
            adj_sparse = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                size=(self.n_wallets, self.n_wallets),
            )

        return node_features, adj_sparse

    def get_flow_summary(self) -> Dict[str, float]:
        """
        High-level summary of on-chain flow direction.

        Returns:
            Dict with:
              - to_exchange: proportion of volume flowing TO exchanges
              - from_exchange: proportion of volume flowing FROM exchanges
              - net_flow: positive = accumulation, negative = distribution
        """
        to_exchange = 0.0
        from_exchange = 0.0

        with self._lock:
            for src, dst, weight in self.edges:
                if dst in self._exchange_indices:
                    to_exchange += weight
                if src in self._exchange_indices:
                    from_exchange += weight

        total = to_exchange + from_exchange + 1e-10
        return {
            "to_exchange": to_exchange / total,
            "from_exchange": from_exchange / total,
            "net_flow": (from_exchange - to_exchange) / total,  # Positive = money leaving exchanges (bullish)
        }

    @property
    def is_connected(self) -> bool:
        return self._running

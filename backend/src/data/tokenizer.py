"""
Financial Tokenizer - Discretize Continuous Market Data into Tokens
====================================================================
Converts continuous float indicator values into discrete token IDs
using quantile-based binning for Language Model training.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"


class FinancialTokenizer:
    """
    Tokenizer for financial time series data.
    
    Converts continuous indicator values to discrete token IDs
    using quantile-based binning (equal frequency bins).
    
    Vocabulary Structure:
    - Each feature has K bins (default 256)
    - Token ID = feature_offset + bin_index
    - Special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3
    """
    
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3
    }
    N_SPECIAL = 4
    
    def __init__(
        self,
        n_bins: int = 256,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            n_bins: Number of bins per feature (default 256 for byte-level)
            feature_columns: List of feature column names to tokenize
        """
        self.n_bins = n_bins
        self.feature_columns = feature_columns or []
        
        # Quantile edges for each feature
        self.bin_edges: Dict[str, np.ndarray] = {}
        
        # Feature to token offset mapping
        self.feature_offsets: Dict[str, int] = {}
        
        # Vocabulary
        self.vocab_size: int = self.N_SPECIAL
        
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Fit the tokenizer on training data.
        
        Computes quantile bin edges for each feature.
        
        Args:
            df: Training DataFrame with features
            feature_columns: Optional list of columns to tokenize
        """
        if feature_columns:
            self.feature_columns = feature_columns
        elif not self.feature_columns:
            # Auto-detect numeric columns (exclude metadata)
            exclude_cols = {"date", "ticker", "target", "volume"}
            self.feature_columns = [
                c for c in df.columns
                if c not in exclude_cols and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
        
        logger.info(f"Fitting tokenizer on {len(self.feature_columns)} features")
        
        # Compute bin edges for each feature
        current_offset = self.N_SPECIAL
        
        for col in self.feature_columns:
            values = df[col].drop_nulls().to_numpy().astype(np.float64)
            
            if len(values) == 0:
                logger.warning(f"Skipping {col}: no valid values")
                continue
            
            # Compute quantile edges (n_bins + 1 edges)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(values, quantiles)
            
            # Handle edge cases where all values are same
            if np.all(edges == edges[0]):
                edges = np.array([edges[0] - 1, edges[0] + 1])
            
            self.bin_edges[col] = edges
            self.feature_offsets[col] = current_offset
            current_offset += self.n_bins
        
        self.vocab_size = current_offset
        self.is_fitted = True
        
        logger.info(f"Tokenizer fitted. Vocabulary size: {self.vocab_size}")
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform features to token IDs.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with token ID columns (integer)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before transform")
        
        token_columns = []
        
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, using [UNK]")
                token_col = pl.lit(self.SPECIAL_TOKENS["[UNK]"]).alias(f"{col}_token")
                token_columns.append(token_col)
                continue
            
            values = df[col].to_numpy().astype(np.float64)
            edges = self.bin_edges[col]
            offset = self.feature_offsets[col]
            
            # Digitize: returns bin indices 1 to n_bins, we want 0 to n_bins-1
            bin_indices = np.digitize(values, edges[1:-1])  # Skip first and last edge
            
            # Handle NaN -> [UNK]
            bin_indices = np.where(np.isnan(values), self.SPECIAL_TOKENS["[UNK]"], bin_indices)
            
            # Add offset to get final token IDs
            token_ids = (offset + bin_indices).astype(np.int32)
            
            token_columns.append(pl.Series(f"{col}_token", token_ids))
        
        return df.with_columns(token_columns)
    
    def fit_transform(self, df: pl.DataFrame, feature_columns: Optional[List[str]] = None) -> pl.DataFrame:
        """Fit and transform in one call."""
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def encode_sequence(self, token_df: pl.DataFrame) -> np.ndarray:
        """
        Convert tokenized DataFrame to a flat sequence of token IDs.
        
        For language modeling, we interleave all feature tokens per timestep.
        
        Args:
            token_df: DataFrame with *_token columns
            
        Returns:
            1D array of token IDs [seq_len * n_features]
        """
        token_cols = [c for c in token_df.columns if c.endswith("_token")]
        
        # Stack all token columns [timesteps, n_features]
        tokens = token_df.select(token_cols).to_numpy().astype(np.int32)
        
        # Flatten row-wise (each timestep's features in sequence)
        return tokens.flatten()
    
    def save(self, path: Path = TOKENIZER_PATH):
        """Save tokenizer to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "n_bins": self.n_bins,
            "feature_columns": self.feature_columns,
            "bin_edges": self.bin_edges,
            "feature_offsets": self.feature_offsets,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path = TOKENIZER_PATH) -> "FinancialTokenizer":
        """Load tokenizer from pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        tokenizer = cls(n_bins=state["n_bins"])
        tokenizer.feature_columns = state["feature_columns"]
        tokenizer.bin_edges = state["bin_edges"]
        tokenizer.feature_offsets = state["feature_offsets"]
        tokenizer.vocab_size = state["vocab_size"]
        tokenizer.is_fitted = True
        
        logger.info(f"Tokenizer loaded from {path}. Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    
    def decode_token(self, token_id: int) -> Tuple[str, int]:
        """
        Decode a single token ID to (feature_name, bin_index).
        
        Args:
            token_id: Token ID
            
        Returns:
            Tuple of (feature_name, bin_index) or ("[SPECIAL]", token_id)
        """
        # Check special tokens
        for name, tid in self.SPECIAL_TOKENS.items():
            if token_id == tid:
                return (name, token_id)
        
        # Find which feature this token belongs to
        for col in self.feature_columns:
            offset = self.feature_offsets[col]
            if offset <= token_id < offset + self.n_bins:
                return (col, token_id - offset)
        
        return ("[UNK]", token_id)


def tokenize_all_parquet(
    input_dir: Path = DATA_PROCESSED_DIR,
    output_dir: Path = DATA_TOKENIZED_DIR,
    tokenizer_path: Path = TOKENIZER_PATH,
    n_bins: int = 256
):
    """
    Tokenize all processed parquet files.
    
    Args:
        input_dir: Directory with processed parquet files
        output_dir: Directory for tokenized output
        tokenizer_path: Path to save tokenizer
        n_bins: Number of bins per feature
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return
    
    # Load all data for fitting
    logger.info(f"Loading {len(parquet_files)} parquet files for tokenizer fitting...")
    all_dfs = [pl.read_parquet(f) for f in parquet_files]
    combined_df = pl.concat(all_dfs)
    
    # Fit tokenizer
    tokenizer = FinancialTokenizer(n_bins=n_bins)
    tokenizer.fit(combined_df)
    tokenizer.save(tokenizer_path)
    
    # Transform each file
    for file_path in parquet_files:
        df = pl.read_parquet(file_path)
        df_tokenized = tokenizer.transform(df)
        
        # Save tokenized data
        output_path = output_dir / file_path.name
        df_tokenized.write_parquet(output_path)
        logger.info(f"Tokenized {file_path.stem} -> {output_path}")
    
    logger.info(f"Tokenization complete. Vocab size: {tokenizer.vocab_size}")


def main():
    """CLI entry point."""
    tokenize_all_parquet()


if __name__ == "__main__":
    main()

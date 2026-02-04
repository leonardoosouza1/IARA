"""
Dataset - Stochastic Shuffling & Target-Only Loss Masking
==========================================================
Refactored dataset to prevent structural overfitting.
1. Shuffles feature tokens randomly per timestep.
2. Places Target Token (Price Return) always at the end.
3. Masks loss so model only learns to predict Price.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.tokenizer import FinancialTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"


class StructuredForexDataset(Dataset):
    """
    Dataset that treats each timestep as a "Bag of Features" + Target.
    
    Structure per timestep: [Feature_1, Feature_2, ..., Feature_N, TARGET]
    
    - Features are randomly shuffled during training to prevent memorizing order.
    - Target is always last.
    - Loss is masked to only penalize Target prediction.
    """
    
    def __init__(
        self,
        data_dir: Path = DATA_TOKENIZED_DIR,
        tokenizer_path: Path = TOKENIZER_PATH,
        seq_len: int = 384,
        stride: int = 128,
        tickers: Optional[List[str]] = None,
        is_training: bool = True,
        target_column: str = "close_log_ret"
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.is_training = is_training
        self.target_column = target_column
        
        # Load tokenizer to identify columns
        self.tokenizer = FinancialTokenizer.load(tokenizer_path)
        self.feature_columns = self.tokenizer.feature_columns
        
        # Identify Target and Feature indices
        self.target_token_col = f"{target_column}_token"
        try:
            # We need to find the index of the target column in the loaded arrays
            # But we load files individually. 
            pass 
        except ValueError:
            logger.warning(f"Target column {target_column} not found in tokenizer!")
            
        # Load data as list of 2D arrays [timesteps, n_cols]
        # We store (features_array, target_array) per ticker
        self.samples = self._load_data(tickers)
        
        # Compute valid starting positions
        self.indices = self._compute_indices()
        
        logger.info(
            f"StructuredDataset initialized: {len(self.indices)} windows. "
            f"Training={is_training}, Target={target_column}"
        )

    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load data into memory.
        Returns list of (features_matrix, target_vector).
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
            
        data_list = []
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                
                # Check for target column
                if self.target_token_col not in df.columns:
                    logger.warning(f"Skipping {file_path.name}: Target {self.target_token_col} not found")
                    continue
                
                # Get all token columns
                token_cols = [c for c in df.columns if c.endswith("_token")]
                
                # Separate Features vs Target
                feat_cols = [c for c in token_cols if c != self.target_token_col]
                
                # Extract arrays
                features = df.select(feat_cols).to_numpy().astype(np.int32) # [T, N_features]
                targets = df.select(self.target_token_col).to_numpy().astype(np.int32).flatten() # [T]
                
                # Validate length
                if len(features) > self.seq_len:
                    data_list.append((features, targets))
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return data_list

    def _compute_indices(self) -> List[Tuple[int, int]]:
        """Compute (sample_idx, start_row) for sliding windows."""
        indices = []
        
        # We need N tokens. 
        # Each timestep produces (N_features + 1) tokens.
        # Let's define context window in terms of TIMESTEPS (Rows), not tokens?
        # The user specified seq_len in tokens usually (e.g. 512).
        # But our "tokens" are now dynamic. 
        # Actually, let's keep seq_len as "Number of Tokens".
        # 
        # If we have F features + 1 target per row.
        # Tokens per row = F + 1.
        # Rows per window = seq_len // (F + 1).
        
        if not self.samples:
            return []
            
        n_features = self.samples[0][0].shape[1]
        tokens_per_row = n_features + 1
        self.rows_per_window = max(1, self.seq_len // tokens_per_row)
        
        # Update actual seq_len to be multiple of tokens_per_row to avoid partial rows
        self.actual_seq_len = self.rows_per_window * tokens_per_row
        
        for i, (feats, _) in enumerate(self.samples):
            n_rows = len(feats)
            # Need rows_per_window + 1 (for target shifting)
            # Wait, if we predict target at current step, we technically need next step's target?
            # No, standard LM: [A, B, C] -> Predict [B, C, D]
            # So we need 1 extra token.
            # Easiest is to take 1 extra ROW to be safe.
            
            max_start = n_rows - self.rows_per_window - 1
            if max_start < 0:
                continue
                
            for start in range(0, max_start, max(1, self.stride // tokens_per_row)):
                indices.append((i, start))
                
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx, start_row = self.indices[idx]
        features_all, targets_all = self.samples[sample_idx]
        
        # Slice window
        end_row = start_row + self.rows_per_window
        
        # Get features and targets for this window
        # Plus one extra row for the shift (next token prediction)
        # Actually, we construct the sequence of length N, and targets is shifted.
        # So we need data for N+1 tokens. 
        # Since we block by row, we take 1 extra row and truncate later if needed.
        
        window_feats = features_all[start_row : end_row + 1] # [W+1, F]
        window_targets = targets_all[start_row : end_row + 1] # [W+1]
        
        # Construct flat sequence
        input_tokens = []
        target_labels = [] # For loss calculation
        
        n_features = window_feats.shape[1]
        
        # Iterate through rows
        # We need to build exactly `actual_seq_len` inputs.
        
        for i in range(len(window_feats)):
            row_feats = window_feats[i]
            row_target = window_targets[i]
            
            # 1. Shuffle Features (if training)
            if self.is_training:
                # Random permutation of indices
                perm = np.random.permutation(n_features)
                row_feats_shuffled = row_feats[perm]
            else:
                row_feats_shuffled = row_feats # Fixed order for val
            
            # 2. Append to sequence: [F1, F2... Fn, T]
            input_tokens.extend(row_feats_shuffled)
            input_tokens.append(row_target)
            
            # 3. Build Labels (Masking)
            # Input:  [F1, F2, ..., Fn, T]
            # Label:  [F2, F3, ..., T,  Next_F1]
            
            # We want loss ONLY when predicting T.
            # So label corresponding to input Fn should be T.
            # Label corresponding to input T should be Next_F1 (or ignored).
            # User said: "Gradient only based on price movement".
            # So we ignore prediction of Next_F1 too?
            # "The gradient should only update weights based on the model's ability to predict the Price Movement."
            # This implies P(Target | Features).
            
            # Mask for Feature Inputs (predicting next feature or target)
            # F1 -> predict F2 (Ignore)
            # ...
            # Fn-1 -> predict Fn (Ignore)
            # Fn -> predict T (KEEP!)
            # T -> predict Next_F1 (Ignore)
            
            # So we append -100 for all features except the last one.
            # Actually, the labels array aligns with inputs.
            # input[k] predicts input[k+1].
            # input[LAST_FEAT_INDEX] predicts input[TARGET_INDEX] = T.
            # So label for input[LAST_FEAT_INDEX] is T.
            
            # Let's add -100 for (n_features - 1) times
            # Then add T for 1 time (corresponding to Fn)
            # Then add -100 for 1 time (corresponding to T predicting next)
            
            # Wait, labels are aligned with INPUTS.
            # If input is F1, label is F2. Mask -100.
            # ...
            # If input is Fn, label is T. Keep T.
            # If input is T, label is Next_F1. Mask -100.
            
            # So for each row:
            # - n_features inputs are features.
            # - The last feature input (Fn) predicts T.
            # - The Target input (T) predicts next.
            
            # Labels mapping corresponding to [F1...Fn, T]:
            # F1 -> -100
            # ...
            # Fn -> T (The actual target value)
            # T  -> -100
            
            labels_row = [-100] * (n_features - 1)
            labels_row.append(row_target) # Corresponds to Fn input
            labels_row.append(-100)       # Corresponds to T input
            
            target_labels.extend(labels_row)

        # Truncate to exact seq_len
        # We generated chunks of (F+1).
        # We need input sequence of length seq_len.
        # labels also length seq_len.
        # But standard causal training:
        # inputs = seq[:-1]
        # targets = seq[1:] (aligned with inputs)
        # So we create a sequence of length seq_len + 1
        
        full_seq = np.array(input_tokens[:self.actual_seq_len + 1], dtype=np.int64)
        full_labels = np.array(target_labels[:self.actual_seq_len + 1], dtype=np.int64)
        
        # Shift
        x = torch.tensor(full_seq[:-1])
        y = torch.tensor(full_seq[1:]) # The standard labels (actual next tokens)
        
        # Apply mask to y
        # We constructed `target_labels` to align with `input_tokens`.
        # `target_labels[i]` is the "desired loss target" for `input_tokens[i]`.
        # My logic above: "If input is Fn, label is T".
        # So `target_labels[index_of_Fn]` = T.
        # `y` (standard shifted) will have `T` at `index_of_Fn`.
        # So we can just use `target_labels` logic to build the mask or just use `target_labels` directly as y_masked?
        
        # Let's re-verify:
        # Sequence: [F1, F2, T]
        # Inputs:   [F1, F2, T] (len 3, but usually we feed N and predict N)
        # x: [F1, F2]
        # y: [F2, T ]
        # Masked y: [-100, T]
        
        # My `target_labels` construction:
        # Row: [F1, F2, T]
        # labels_row: [-100, T, -100]
        # labels_row[0] correspond to input F1. (Label should be F2, we set -100)
        # labels_row[1] correspond to input F2. (Label should be T, we set T)
        # labels_row[2] correspond to input T.  (Label should be Next_F1, we set -100)
        
        # So `target_labels` IS exactly `y_masked` but aligned with `input_tokens`.
        # We need `y_masked` corresponding to `x`.
        # `x` is `input_tokens[:-1]`.
        # So `y_masked` is `target_labels[:-1]`.
        
        y_masked = torch.tensor(full_labels[:-1])
        
        return x, y_masked


def create_dataloader(
    data_dir: Path = DATA_TOKENIZED_DIR,
    seq_len: int = 384,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> DataLoader:
    dataset = StructuredForexDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        is_training=shuffle, # Only shuffle features during training
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

def main():
    # Test
    ds = StructuredForexDataset(seq_len=20)
    if len(ds) > 0:
        x, y = ds[0]
        print(f"X: {x.tolist()}")
        print(f"Y: {y.tolist()}")

if __name__ == "__main__":
    main()

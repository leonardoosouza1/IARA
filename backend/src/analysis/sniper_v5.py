"""
Sniper Protocol v5 - High-Conviction Trading Filter
====================================================
Multi-factor filtering to maximize Win Rate and Profit Factor.

Filters:
1. Confidence Filter: prob > (mean + 1.5σ) or prob < (mean - 1.5σ)
2. Volatility Filter: ATR > 25th percentile (avoid dead markets)
3. Session Filter: London (08:00-16:00) and NY (13:00-21:00) overlap

Metrics:
- Win Rate (Target: >56%)
- Profit Factor (Target: >1.5)
- Recovery Factor
- Sniper Trades vs Total Trades
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

# Simulation Parameters
INITIAL_BALANCE = 1000.0
FIXED_RISK = 20.0
RISK_REWARD_RATIO = 1.0

# Sniper Parameters
CONFIDENCE_SIGMA = 1.5
ATR_PERCENTILE = 25
LONDON_START = 8   # 08:00 UTC
LONDON_END = 16    # 16:00 UTC
NY_START = 13      # 13:00 UTC
NY_END = 21        # 21:00 UTC


@dataclass
class TradeResult:
    """Individual trade result."""
    timestamp_idx: int
    hour: int
    prob: float
    predicted_dir: int
    true_dir: int
    won: bool
    profit: float
    is_sniper: bool
    session: str
    atr: float
    confidence_percentile: float


@dataclass
class SniperMetrics:
    """Aggregated sniper protocol metrics."""
    total_trades: int = 0
    sniper_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_profit: float = 0.0
    final_balance: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    avg_yield_per_trade: float = 0.0
    session_win_rates: Dict[str, float] = field(default_factory=dict)
    hourly_win_rates: Dict[int, float] = field(default_factory=dict)


def load_model(device: torch.device) -> IARA:
    """Load trained I.A.R.A. model."""
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    trainer_config = checkpoint.get("config")
    if trainer_config:
        model_config = LiquidConfig(
            input_dim=trainer_config.input_dim,
            hidden_dim=trainer_config.hidden_dim,
            dropout=trainer_config.dropout
        )
    else:
        model_config = LiquidConfig(hidden_dim=512)
    
    model = IARA(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model


def load_full_data(ticker: str = "EURUSD") -> pl.DataFrame:
    """Load full dataset with all columns."""
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    return pl.read_parquet(file_path)


@torch.no_grad()
def generate_all_predictions(
    model: IARA,
    df: pl.DataFrame,
    device: torch.device,
    seq_len: int = 64
) -> List[Dict]:
    """Generate predictions with full context."""
    model.eval()
    
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    
    # Get metadata
    has_hour = "hour" in df.columns
    has_atr = "atr" in df.columns or "atr_norm" in df.columns
    has_target = "close_log_ret" in df.columns
    
    results = []
    n_samples = len(features) - seq_len
    
    for i in range(n_samples):
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x)
        prob = torch.sigmoid(logits).cpu().item()
        
        target_idx = i + seq_len - 1
        
        result = {
            "idx": target_idx,
            "prob": prob,
        }
        
        if has_hour:
            result["hour"] = df["hour"][target_idx]
        else:
            result["hour"] = 12  # Default
        
        if has_atr:
            atr_col = "atr" if "atr" in df.columns else "atr_norm"
            result["atr"] = df[atr_col][target_idx]
        else:
            result["atr"] = 1.0
        
        if has_target:
            true_ret = df["close_log_ret"][target_idx]
            result["true_dir"] = 1 if true_ret > 0 else 0
        else:
            result["true_dir"] = 0
        
        results.append(result)
    
    return results


def get_session(hour: int) -> str:
    """Determine trading session."""
    london = LONDON_START <= hour < LONDON_END
    ny = NY_START <= hour < NY_END
    
    if london and ny:
        return "London-NY Overlap"
    elif london:
        return "London"
    elif ny:
        return "New York"
    elif 0 <= hour < 8:
        return "Asian"
    else:
        return "Other"


def apply_sniper_filters(
    predictions: List[Dict],
    prob_mean: float,
    prob_std: float,
    atr_threshold: float,
    top_percentile: float = 10.0
) -> List[Dict]:
    """Apply multi-factor sniper filters."""
    
    # Calculate confidence thresholds
    upper_threshold = prob_mean + CONFIDENCE_SIGMA * prob_std
    lower_threshold = prob_mean - CONFIDENCE_SIGMA * prob_std
    
    logger.info(f"Confidence thresholds: UP > {upper_threshold:.4f}, DOWN < {lower_threshold:.4f}")
    logger.info(f"ATR threshold (25th pctl): {atr_threshold:.6f}")
    
    for pred in predictions:
        prob = pred["prob"]
        atr = pred["atr"]
        hour = pred["hour"]
        
        # Determine if sniper trade
        confidence_pass = (prob > upper_threshold) or (prob < lower_threshold)
        volatility_pass = atr > atr_threshold
        session = get_session(hour)
        session_pass = session in ["London-NY Overlap", "London", "New York"]
        
        # All filters must pass
        pred["is_sniper"] = confidence_pass and volatility_pass and session_pass
        pred["session"] = session
        
        # Confidence percentile (for ranking)
        if prob > prob_mean:
            pred["confidence_percentile"] = (prob - prob_mean) / prob_std
        else:
            pred["confidence_percentile"] = (prob_mean - prob) / prob_std
    
    return predictions


def simulate_sniper_trades(
    predictions: List[Dict],
    sniper_only: bool = True
) -> Tuple[List[TradeResult], SniperMetrics]:
    """Simulate trading with sniper protocol."""
    
    balance = INITIAL_BALANCE
    balances = [balance]
    trades: List[TradeResult] = []
    
    for pred in predictions:
        if sniper_only and not pred["is_sniper"]:
            continue
        
        prob = pred["prob"]
        true_dir = pred["true_dir"]
        
        # Determine trade direction
        if prob > 0.5:
            predicted_dir = 1  # BUY
        else:
            predicted_dir = 0  # SELL
        
        # Trade outcome
        won = predicted_dir == true_dir
        profit = FIXED_RISK * RISK_REWARD_RATIO if won else -FIXED_RISK
        balance += profit
        balances.append(balance)
        
        trade = TradeResult(
            timestamp_idx=pred["idx"],
            hour=pred["hour"],
            prob=prob,
            predicted_dir=predicted_dir,
            true_dir=true_dir,
            won=won,
            profit=profit,
            is_sniper=pred["is_sniper"],
            session=pred["session"],
            atr=pred["atr"],
            confidence_percentile=pred["confidence_percentile"]
        )
        trades.append(trade)
    
    # Calculate metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = n_trades - wins
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    
    gross_profit = sum(t.profit for t in trades if t.profit > 0)
    gross_loss = abs(sum(t.profit for t in trades if t.profit < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    net_profit = balance - INITIAL_BALANCE
    
    # Max Drawdown
    balances_array = np.array(balances)
    peak = np.maximum.accumulate(balances_array)
    drawdown = (peak - balances_array) / peak
    max_drawdown = drawdown.max() * 100
    
    # Recovery Factor
    recovery_factor = net_profit / (max_drawdown * INITIAL_BALANCE / 100) if max_drawdown > 0 else 0
    
    # Avg Yield per Trade
    avg_yield = net_profit / n_trades if n_trades > 0 else 0
    
    # Session Win Rates
    session_stats: Dict[str, List[bool]] = {}
    for t in trades:
        if t.session not in session_stats:
            session_stats[t.session] = []
        session_stats[t.session].append(t.won)
    
    session_win_rates = {
        s: sum(wins) / len(wins) * 100 if len(wins) > 0 else 0
        for s, wins in session_stats.items()
    }
    
    # Hourly Win Rates
    hourly_stats: Dict[int, List[bool]] = {}
    for t in trades:
        if t.hour not in hourly_stats:
            hourly_stats[t.hour] = []
        hourly_stats[t.hour].append(t.won)
    
    hourly_win_rates = {
        h: sum(wins) / len(wins) * 100 if len(wins) > 0 else 0
        for h, wins in hourly_stats.items()
    }
    
    sniper_count = sum(1 for t in trades if t.is_sniper)
    
    metrics = SniperMetrics(
        total_trades=len(predictions),
        sniper_trades=sniper_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        net_profit=net_profit,
        final_balance=balance,
        max_drawdown=max_drawdown,
        recovery_factor=recovery_factor,
        avg_yield_per_trade=avg_yield,
        session_win_rates=session_win_rates,
        hourly_win_rates=hourly_win_rates
    )
    
    return trades, metrics


def generate_visualizations(
    predictions: List[Dict],
    trades: List[TradeResult],
    metrics: SniperMetrics
):
    """Generate analysis visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Probability Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    probs = [p["prob"] for p in predictions]
    prob_mean = np.mean(probs)
    prob_std = np.std(probs)
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(probs, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(prob_mean, color='red', linestyle='--', label=f'Mean: {prob_mean:.3f}')
    ax1.axvline(prob_mean + 1.5 * prob_std, color='green', linestyle=':', label=f'+1.5σ: {prob_mean + 1.5 * prob_std:.3f}')
    ax1.axvline(prob_mean - 1.5 * prob_std, color='orange', linestyle=':', label=f'-1.5σ: {prob_mean - 1.5 * prob_std:.3f}')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Model Probability Distribution (Confidence Tails)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sniper vs Regular trades
    ax2 = axes[1]
    sniper_probs = [p["prob"] for p in predictions if p["is_sniper"]]
    regular_probs = [p["prob"] for p in predictions if not p["is_sniper"]]
    
    ax2.hist(regular_probs, bins=30, alpha=0.5, label=f'Regular ({len(regular_probs)})', color='gray')
    ax2.hist(sniper_probs, bins=30, alpha=0.7, label=f'Sniper ({len(sniper_probs)})', color='green')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sniper vs Regular Trade Probabilities')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sniper_probability_distribution.png', dpi=150)
    plt.close()
    logger.info(f"Saved probability distribution to {OUTPUT_DIR / 'sniper_probability_distribution.png'}")
    
    # 2. Win Rate Heatmap by Hour
    fig, ax = plt.subplots(figsize=(12, 4))
    
    hours = list(range(24))
    win_rates = [metrics.hourly_win_rates.get(h, 50) for h in hours]
    trade_counts = [sum(1 for t in trades if t.hour == h) for h in hours]
    
    # Create heatmap data
    colors = ['red' if wr < 50 else 'green' for wr in win_rates]
    bars = ax.bar(hours, win_rates, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax.axhline(56, color='gold', linestyle='-', linewidth=2, label='Target: 56%')
    
    # Add trade count annotation
    for i, (bar, count) in enumerate(zip(bars, trade_counts)):
        if count > 0:
            ax.annotate(f'{count}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, color='gray')
    
    # Highlight session overlaps
    ax.axvspan(LONDON_START, LONDON_END, alpha=0.1, color='blue', label='London')
    ax.axvspan(NY_START, NY_END, alpha=0.1, color='orange', label='New York')
    
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Hour of Day (Sniper Trades)')
    ax.set_xticks(hours)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sniper_hourly_winrate.png', dpi=150)
    plt.close()
    logger.info(f"Saved hourly win rate to {OUTPUT_DIR / 'sniper_hourly_winrate.png'}")


def run_sniper_analysis():
    """Run the full Sniper Protocol analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and data
    model = load_model(device)
    df = load_full_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_all_predictions(model, df, device)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Calculate statistics
    probs = np.array([p["prob"] for p in predictions])
    prob_mean = probs.mean()
    prob_std = probs.std()
    logger.info(f"Prob stats: mean={prob_mean:.4f}, std={prob_std:.4f}")
    
    # ATR threshold
    atrs = np.array([p["atr"] for p in predictions])
    atr_threshold = np.percentile(atrs, ATR_PERCENTILE)
    
    # Apply sniper filters
    predictions = apply_sniper_filters(predictions, prob_mean, prob_std, atr_threshold)
    
    sniper_count = sum(1 for p in predictions if p["is_sniper"])
    logger.info(f"Sniper trades identified: {sniper_count} / {len(predictions)} ({sniper_count/len(predictions)*100:.1f}%)")
    
    # Simulate trades
    print("\n" + "=" * 80)
    print("SNIPER PROTOCOL v5 ANALYSIS")
    print("=" * 80)
    
    # 1. All trades (baseline)
    trades_all, metrics_all = simulate_sniper_trades(predictions, sniper_only=False)
    print(f"\n📊 BASELINE (All Trades):")
    print(f"   Trades: {metrics_all.sniper_trades}")
    print(f"   Win Rate: {metrics_all.win_rate:.1f}%")
    print(f"   Profit Factor: {metrics_all.profit_factor:.2f}")
    print(f"   Net Profit: ${metrics_all.net_profit:.2f}")
    
    # 2. Sniper trades only
    trades_sniper, metrics_sniper = simulate_sniper_trades(predictions, sniper_only=True)
    print(f"\n🎯 SNIPER PROTOCOL (Filtered Trades):")
    print(f"   Trades: {metrics_sniper.sniper_trades} ({metrics_sniper.sniper_trades/len(predictions)*100:.1f}% of total)")
    print(f"   Win Rate: {metrics_sniper.win_rate:.1f}% {'✓' if metrics_sniper.win_rate > 56 else ''}")
    print(f"   Profit Factor: {metrics_sniper.profit_factor:.2f} {'✓' if metrics_sniper.profit_factor > 1.5 else ''}")
    print(f"   Net Profit: ${metrics_sniper.net_profit:.2f}")
    print(f"   Max Drawdown: {metrics_sniper.max_drawdown:.1f}%")
    print(f"   Recovery Factor: {metrics_sniper.recovery_factor:.2f}")
    print(f"   Avg Yield/Trade: ${metrics_sniper.avg_yield_per_trade:.2f}")
    
    # Session analysis
    print(f"\n📍 SESSION ANALYSIS:")
    for session, wr in sorted(metrics_sniper.session_win_rates.items(), key=lambda x: -x[1]):
        count = sum(1 for t in trades_sniper if t.session == session)
        status = "✓" if wr > 55 else ""
        print(f"   {session:20s}: {wr:5.1f}% ({count} trades) {status}")
    
    # Top 10% highest confidence trades
    print(f"\n🔥 TOP 10% HIGHEST CONFIDENCE:")
    sorted_by_confidence = sorted(predictions, key=lambda x: x["confidence_percentile"], reverse=True)
    top_10_pct = sorted_by_confidence[:len(sorted_by_confidence)//10]
    
    top_trades = []
    for pred in top_10_pct:
        if pred["prob"] > 0.5:
            predicted_dir = 1
        else:
            predicted_dir = 0
        won = predicted_dir == pred["true_dir"]
        top_trades.append(won)
    
    top_win_rate = sum(top_trades) / len(top_trades) * 100 if top_trades else 0
    print(f"   Trades: {len(top_trades)}")
    print(f"   Win Rate: {top_win_rate:.1f}%")
    
    # Generate visualizations
    generate_visualizations(predictions, trades_sniper, metrics_sniper)
    
    print("\n" + "=" * 80)
    print("SNIPER SUMMARY")
    print("=" * 80)
    print(f"│ Metric           │ Baseline │ Sniper   │ Target   │ Status   │")
    print(f"├──────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"│ Win Rate         │ {metrics_all.win_rate:6.1f}%  │ {metrics_sniper.win_rate:6.1f}%  │  >56%    │ {'✓' if metrics_sniper.win_rate > 56 else '✗':^8} │")
    print(f"│ Profit Factor    │ {metrics_all.profit_factor:6.2f}   │ {metrics_sniper.profit_factor:6.2f}   │   >1.5   │ {'✓' if metrics_sniper.profit_factor > 1.5 else '✗':^8} │")
    print(f"│ Net Profit       │ ${metrics_all.net_profit:>7.0f} │ ${metrics_sniper.net_profit:>7.0f} │    -     │    -     │")
    print(f"│ Trades           │ {metrics_all.sniper_trades:>8} │ {metrics_sniper.sniper_trades:>8} │    -     │    -     │")
    print(f"└──────────────────┴──────────┴──────────┴──────────┴──────────┘")
    
    print(f"\n📁 Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_sniper_analysis()

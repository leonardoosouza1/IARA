import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def _run(cmd: list[str], cwd: str | None = None, log_file: Path | None = None) -> None:
    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(cmd, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, cwd=cwd, check=True)


def _load_backtest_metrics(report_path: Path) -> dict:
    """Extract key metrics from backtest report for gating decisions."""
    if not report_path.exists():
        return {}
    with open(report_path, "r") as f:
        report = json.load(f)
    # Backtest report stores metrics at top level, not under "summary"
    return {
        "realized_pnl": report.get("pnl_realized", 0.0),
        "unrealized_pnl": 0.0,  # Not tracked in current report
        "total_trades": report.get("trade_count", 0),
        "winning_trades": 0,  # Not tracked in current report
        "losing_trades": 0,  # Not tracked in current report
        "max_drawdown": 0.0,  # Not tracked in current report
    }


def _is_better_than_previous(
    current: dict, previous: dict | None, min_trades: int = 5
) -> bool:
    """Determine if current cycle is better than previous for model promotion."""
    if previous is None:
        # First cycle: always promote to establish a baseline
        return True

    # Must have minimum trades to be statistically significant
    if current["total_trades"] < min_trades:
        return False

    # Primary: realized PnL must be better
    if current["realized_pnl"] > previous["realized_pnl"]:
        return True

    # Secondary: if PnL similar, check drawdown (lower is better)
    pnl_diff = abs(current["realized_pnl"] - previous["realized_pnl"])
    if pnl_diff < 10:  # Within $10
        return current["max_drawdown"] < previous["max_drawdown"]

    return False


def _promote_model(model_path: Path, cycle_dir: Path, cycle: int) -> Path:
    """Promote model to best_model_v9.pth with backup."""
    # Backup previous best if exists
    if model_path.exists():
        backup_path = cycle_dir / f"previous_best_cycle{cycle:02d}.pth"
        import shutil

        shutil.copy2(model_path, backup_path)

    # The trained model is already at model_path (train_v9 writes there)
    # Just verify it exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-Learning Loop for IARA v9 (train -> backtest -> replay -> train)"
    )
    parser.add_argument(
        "--cycles", type=int, default=3, help="Number of train/backtest cycles"
    )
    parser.add_argument(
        "--train_epochs", type=int, default=10, help="Epochs per training cycle"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--train_seq_len", type=int, default=64, help="Sequence length for training"
    )
    parser.add_argument(
        "--train_lr", type=float, default=0.0005, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_patience",
        type=int,
        default=None,
        help="Early stopping patience for training (defaults to max(5, train_epochs//2))",
    )
    parser.add_argument(
        "--train_worker_hidden", type=int, default=64, help="Worker hidden dim"
    )
    parser.add_argument(
        "--train_coord_hidden", type=int, default=256, help="Coordinator hidden dim"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume training from existing checkpoint (default: true)",
    )
    parser.add_argument(
        "--no_amp", action="store_true", default=False, help="Disable AMP in training"
    )
    parser.add_argument(
        "--require_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA/GPU to run (default: true)",
    )

    parser.add_argument(
        "--backtest_data",
        type=str,
        default="data/processed/EURUSD.parquet",
        help="Parquet/CSV path used by backtest",
    )
    parser.add_argument("--backtest_start_date", type=str, default="2023-01-01")
    parser.add_argument("--backtest_end_date", type=str, default=None)
    parser.add_argument("--backtest_speed", type=float, default=200.0)
    parser.add_argument("--backtest_max_steps", type=int, default=2000)
    parser.add_argument(
        "--gating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic model promotion based on backtest quality (default: true)",
    )
    parser.add_argument(
        "--min_trades_for_promotion",
        type=int,
        default=5,
        help="Minimum trades required in backtest for model promotion",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="logs/auto_learn_v9",
        help="Folder to store per-cycle reports",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="logs/models/best_model_v9.pth",
        help="Path to the model checkpoint to evaluate/train (train_v9 writes to this path)",
    )

    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for auto_learn_v9, but torch.cuda.is_available() is False"
        )

    project_root = str(Path(__file__).resolve().parents[2])
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)

    summary: list[dict] = []

    for cycle in range(1, args.cycles + 1):
        cycle_dir = work_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # 1) TRAIN
        train_cmd = [
            sys.executable,
            "-u",
            "src/brain/train_v9.py",
            "--epochs",
            str(args.train_epochs),
            "--lr",
            str(args.train_lr),
            "--batch_size",
            str(args.train_batch_size),
            "--seq_len",
            str(args.train_seq_len),
            "--worker_hidden",
            str(args.train_worker_hidden),
            "--coord_hidden",
            str(args.train_coord_hidden),
        ]
        patience = args.train_patience
        if patience is None:
            patience = max(5, args.train_epochs // 2)
        train_cmd.extend(["--patience", str(int(patience))])

        if args.resume:
            train_cmd.append("--resume")
        if args.no_amp:
            train_cmd.append("--no_amp")

        t0 = time.time()
        _run(train_cmd, cwd=project_root, log_file=cycle_dir / "train.log")
        train_s = time.time() - t0

        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model checkpoint at {model_path}, but it was not created."
            )

        # 2) BACKTEST (generates captures: errors/success/patterns)
        report_path = cycle_dir / "backtest_report.json"
        backtest_cmd = [
            sys.executable,
            "-u",
            "src/brain/backtest_v9.py",
            "--model",
            str(model_path),
            "--data",
            args.backtest_data,
            "--start_date",
            args.backtest_start_date,
            "--speed",
            str(args.backtest_speed),
            "--max_steps",
            str(args.backtest_max_steps),
            "--report_path",
            str(report_path),
            "--worker_hidden",
            str(args.train_worker_hidden),
            "--coord_hidden",
            str(args.train_coord_hidden),
            "--seq_len",
            str(args.train_seq_len),
            "--no_sleep",
        ]
        if args.backtest_end_date:
            backtest_cmd.extend(["--end_date", args.backtest_end_date])

        t1 = time.time()
        _run(backtest_cmd, cwd=project_root, log_file=cycle_dir / "backtest.log")
        backtest_s = time.time() - t1

        # 3) GATING: Evaluate and decide on model promotion
        current_metrics = _load_backtest_metrics(report_path)
        promoted = False
        rollback_to = None

        if args.gating and current_metrics:
            previous_metrics = summary[-1]["metrics"] if summary else None

            if _is_better_than_previous(
                current_metrics, previous_metrics, args.min_trades_for_promotion
            ):
                # Promote: model already at model_path, just backup previous
                promoted_path = _promote_model(model_path, cycle_dir, cycle)
                promoted = True
                print(
                    f"[Cycle {cycle}] Model PROMOTED - PnL: ${current_metrics['realized_pnl']:.2f}, Trades: {current_metrics['total_trades']}"
                )
            else:
                # Rollback: restore previous best if available
                if previous_metrics and len(summary) > 0:
                    prev_cycle_dir = work_dir / f"cycle_{summary[-1]['cycle']:02d}"
                    backup_candidates = list(
                        prev_cycle_dir.glob("previous_best_cycle*.pth")
                    )
                    if backup_candidates:
                        rollback_to = backup_candidates[0]
                    elif (prev_cycle_dir / "model_final.pth").exists():
                        rollback_to = prev_cycle_dir / "model_final.pth"

                    if rollback_to and rollback_to.exists():
                        import shutil

                        shutil.copy2(rollback_to, model_path)
                        print(
                            f"[Cycle {cycle}] Model REJECTED - Rolling back to {rollback_to.name}"
                        )
                        print(
                            f"  Current PnL: ${current_metrics['realized_pnl']:.2f} vs Previous: ${previous_metrics['realized_pnl']:.2f}"
                        )
                    else:
                        print(
                            f"[Cycle {cycle}] Model REJECTED but no rollback candidate found (keeping current)"
                        )
                else:
                    print(
                        f"[Cycle {cycle}] Model REJECTED (first cycle not profitable enough)"
                    )
        else:
            # Gating disabled or no metrics - always promote
            promoted = True
            print(
                f"[Cycle {cycle}] Gating disabled - Model kept (PnL: ${current_metrics.get('realized_pnl', 0):.2f})"
            )

        backtest_report = {}
        if report_path.exists():
            with open(report_path, "r") as f:
                backtest_report = json.load(f)

        entry = {
            "cycle": cycle,
            "train_seconds": float(train_s),
            "backtest_seconds": float(backtest_s),
            "model_path": str(model_path),
            "promoted": promoted,
            "rollback_source": str(rollback_to) if rollback_to else None,
            "metrics": current_metrics,
            "backtest": backtest_report,
        }
        summary.append(entry)

        with open(cycle_dir / "cycle_summary.json", "w") as f:
            json.dump(entry, f, indent=2)

    with open(work_dir / "summary.json", "w") as f:
        json.dump({"cycles": summary}, f, indent=2)


if __name__ == "__main__":
    main()

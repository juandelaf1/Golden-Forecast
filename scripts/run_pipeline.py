#!/usr/bin/env python3
"""
Golden Forecast - Pipeline Orchestrator

Runs the full ML pipeline: preprocessing -> features -> classification -> regression -> validation.
Idempotent: validates outputs before executing each step.
Usage: python scripts/run_pipeline.py [--skip-prepro] [--skip-features] [--skip-classification] [--skip-regression]
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import yaml

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "config" / "pipeline.yaml"


def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def run_step(name: str, script: str, check_output: str = None, skip: bool = False) -> bool:
    """Run a pipeline step if its output does not exist and it is not skipped."""
    if skip:
        print(f"\n{'='*60}\n⏭️  {name} - SKIPPED\n{'='*60}")
        return True

    if check_output and Path(check_output).exists():
        print(f"\n{'='*60}\n✅ {name} - YA EXISTE ({check_output})\n{'='*60}")
        return True

    print(f"\n{'='*60}\n🚀 {name}\n{'='*60}")
    start = datetime.now()

    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800  # 30 min max por paso
    )

    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        print(f"❌ {name} FALLÓ ({elapsed:.1f}s)")
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return False

    if check_output and not Path(check_output).exists():
        print(f"⚠️  {name} terminó pero no generó {check_output}")
        return False

    print(f"✅ {name} OK ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Golden Forecast Pipeline Orchestrator")
    parser.add_argument("--skip-prepro", action="store_true", help="Saltar preprocesamiento")
    parser.add_argument("--skip-features", action="store_true", help="Saltar feature engineering")
    parser.add_argument("--skip-classification", action="store_true", help="Saltar clasificación")
    parser.add_argument("--skip-regression", action="store_true", help="Saltar regresión")
    parser.add_argument("--force", action="store_true", help="Forzar re-ejecución ignorando outputs existentes")
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]

    print("="*60)
    print("GOLDEN FORECAST - PIPELINE COMPLETA")
    print("="*60)

    steps = [
        ("PREPROCESAMIENTO (Gema)", "src/preprocessing.py",
         None if args.force else ROOT / data_cfg["processed_dir"] / data_cfg["clean_file"],
         args.skip_prepro),

        ("FEATURE ENGINEERING (Gema)", "src/feature_engineering.py",
         None if args.force else ROOT / data_cfg["processed_dir"] / data_cfg["features_file"],
         args.skip_features),

        ("CLASIFICACIÓN (Joel) — modelos ya entrenados, retrain con --force", "src/models/train.py",
         None if args.force else ROOT / config["models"]["dir"] / "evaluation_results.json",
         args.skip_classification or not args.force),

        ("REGRESIÓN (Juan)", "src/regression.py",
         None if args.force else ROOT / config["models"]["dir"] / config["regression"]["cache_file"],
         args.skip_regression),
    ]

    all_ok = True
    for name, script, output, skip in steps:
        if not run_step(name, script, str(output) if output else None, skip):
            all_ok = False
            break

    if all_ok:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETA - TODOS LOS PASOS OK")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("💥 PIPELINE FALLÓ")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
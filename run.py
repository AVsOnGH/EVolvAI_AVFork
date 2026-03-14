#!/usr/bin/env python3
"""
EVolvAI – single entry point.

Usage:
    python run.py mock        Generate mock demand tensor for async handoff
    python run.py train       Train the GCD-VAE model
    python run.py generate    Generate counterfactual scenarios from trained model
    python run.py all         Run the full pipeline (mock → train → generate)
"""

import sys


def _usage():
    print(__doc__.strip())
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        _usage()

    cmd = sys.argv[1].lower()

    if cmd == "mock":
        from generative_model.mock import save_mock
        save_mock()

    elif cmd == "train":
        from generative_model.train import train
        train()

    elif cmd == "generate":
        from generative_model.generate import generate_all_scenarios
        generate_all_scenarios()

    elif cmd == "all":
        from generative_model.mock import save_mock
        from generative_model.train import train
        from generative_model.generate import generate_all_scenarios

        print("=" * 60)
        print("  STEP 1/3 – Mock Output for Async Handoff")
        print("=" * 60)
        save_mock()

        print("\n" + "=" * 60)
        print("  STEP 2/3 – Training GCD-VAE")
        print("=" * 60)
        model, device = train()

        print("\n" + "=" * 60)
        print("  STEP 3/3 – Generating Counterfactual Scenarios")
        print("=" * 60)
        generate_all_scenarios(model=model, device=device)

        print("\n✅ Full pipeline complete. Check output/ for results.")

    else:
        _usage()


if __name__ == "__main__":
    main()

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

    try:
        if cmd == "mock":
            from generative_core.mock import save_mock
            save_mock()

        elif cmd == "train":
            from generative_core.train import train
            train()

        elif cmd == "generate":
            from generative_core.generate import generate_all_scenarios
            generate_all_scenarios()

        elif cmd == "all":
            from generative_core.mock import save_mock
            from generative_core.train import train
            from generative_core.generate import generate_all_scenarios

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

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}", file=sys.stderr)
        print("   Install with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Launcher script for the enhanced Thought Tracer application.
Uses Textual framework for a sophisticated terminal UI.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logitlens_tui.enhanced_app import EnhancedAppLauncher

MODEL_CHOICES = [
    ("Ministral 8B (3-8B-Instruct)", "./Ministral-3-8B-Instruct-2512-BF16"),
    ("Ministral 3B (3-3B-Instruct)", "./Ministral-3-3B-Instruct-2512"),
]


def main():
    """Main entry point for enhanced Thought Tracer."""
    print("Thought Tracer - Enhanced Edition")
    print("=" * 50)

    try:
        print("Starting enhanced terminal interface...")
        launcher = EnhancedAppLauncher(MODEL_CHOICES)
        launcher.run()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

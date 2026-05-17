from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS: dict[str, str] = {
    "train": "fraud_detection.jobs.train",
    "stream": "fraud_detection.jobs.streaming",
    "produce": "fraud_detection.jobs.producer",
    "api": "fraud_detection.jobs.api",
    "ui": "fraud_detection.jobs.ui",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="fraud", description="Fraud detection platform CLI.")
        parser.add_argument("command", choices=COMMANDS)
        parser.print_help()
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown command {command!r}. Choose one of: {choices}")

    sys.argv = [f"fraud-{command}", *sys.argv[2:]]
    module = importlib.import_module(COMMANDS[command])
    module.main()


if __name__ == "__main__":
    main()

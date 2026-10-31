"""Restore source order and select the first (best) H hypothesis per sentence."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_file")
    args = parser.parse_args()
    hypotheses = {}
    with open(args.generation_file, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("H-"):
                identifier, _, text = line.rstrip("\n").split("\t", 2)
                hypotheses.setdefault(int(identifier[2:]), text)
    if not hypotheses:
        parser.error("No H- hypotheses found")
    if sorted(hypotheses) != list(range(len(hypotheses))):
        parser.error("Sentence IDs are not contiguous from zero; check skipped/filtered inputs")
    for identifier in sorted(hypotheses):
        sys.stdout.write(hypotheses[identifier] + "\n")


if __name__ == "__main__":
    main()

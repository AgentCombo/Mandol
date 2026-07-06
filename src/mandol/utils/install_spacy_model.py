"""Explicit installer for Mandol's recommended BM25 spaCy model."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Optional, Sequence


DEFAULT_SPACY_MODEL = "en_core_web_lg"
DISABLED_COMPONENTS = ["parser", "ner", "textcat", "senter"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install a spaCy model for Mandol BM25 English lemmatization and "
            "higher keyword recall."
        )
    )
    parser.add_argument("model_name", nargs="?", default=DEFAULT_SPACY_MODEL)
    args = parser.parse_args(argv)

    model_name = args.model_name
    print(
        f"Mandol: installing spaCy model {model_name} for BM25 English "
        "lemmatization and higher keyword recall...",
        flush=True,
    )

    try:
        spacy_module = import_module("spacy")
        spacy_cli = import_module("spacy.cli")
    except ImportError as exc:
        print(
            "Mandol: spaCy is not installed. Install Mandol's default runtime "
            f"dependencies first. ({exc})",
            flush=True,
        )
        return 1

    try:
        spacy_cli.download(model_name)
        spacy_module.load(model_name, disable=DISABLED_COMPONENTS)
    except (Exception, SystemExit) as exc:
        print(f"Mandol: failed to install spaCy model {model_name}: {exc}", flush=True)
        return 1

    print(f"Mandol: spaCy model {model_name} installed and verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

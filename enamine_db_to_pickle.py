#!/usr/bin/env python3
"""
cxsmiles_to_pickle.py
Read an Enamine .cxsmiles (tab- or whitespace-delimited) file and save a pickle
with just two columns: 'smiles' and 'zinc_id' (renamed from 'id').
"""

import argparse
import gzip
import io
import os
import re
import sys
import pandas as pd


def infer_sep(sample_line: str) -> str:
    """Infer delimiter: prefer tab if present, else treat as arbitrary whitespace."""
    return "\t" if "\t" in sample_line else r"\s+"


def open_maybe_gzip(path: str, mode: str = "rt"):
    """Open plain or .gz file transparently."""
    if path.endswith(".gz"):
        # Text mode with UTF-8 for header parsing
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def main():
    ap = argparse.ArgumentParser(description="Convert Enamine .cxsmiles to a 2-column pickle (smiles, zinc_id).")
    ap.add_argument("input", help="Path to .cxsmiles file (optionally .gz)")
    ap.add_argument("-o", "--output", default="smiles_zinc_id.pkl",
                    help="Output pickle path (e.g., smiles_zinc_id.pkl or .pkl.gz). Default: %(default)s")
    ap.add_argument("--sep", default=None,
                    help="Delimiter override. If not provided, the script infers (tab or whitespace).")
    ap.add_argument("--chunksize", type=int, default=None,
                    help="Optional: read in chunks (rows per chunk). If omitted, reads in one pass.")
    args = ap.parse_args()

    # Peek at first non-empty line to infer separator if needed
    if args.sep is None:
        with open_maybe_gzip(args.input, "rt") as fh:
            first = fh.readline()
            # If the file might have a BOM or blank lines, skip empties
            while first and not first.strip():
                first = fh.readline()
        if not first:
            print("Error: input file appears to be empty.", file=sys.stderr)
            sys.exit(1)
        sep = infer_sep(first)
    else:
        sep = args.sep

    # We only need these two columns
    usecols = ["smiles", "id"]

    # Build common read_csv kwargs
    read_kwargs = dict(
        sep=sep,
        engine="python",          # robust with regex/whitespace sep
        usecols=usecols,
        dtype={"smiles": "string", "id": "string"},
        na_filter=False,          # keep strings exactly as is
        on_bad_lines="warn" if hasattr(pd, "errors") else "error",
        # If file has a very long header name spacing, pandas still matches by token
    )

    # Read file (optionally with chunks)
    if args.chunksize:
        read_kwargs["chunksize"] = args.chunksize
        chunks = []
        for chunk in pd.read_csv(args.input, **read_kwargs):
            # Rename 'id' -> 'zinc_id' and keep order
            chunk = chunk.rename(columns={"id": "zinc_id"})[["zinc_id", "smiles"]]
            chunks.append(chunk)
        if not chunks:
            print("No data read; check input file/columns.", file=sys.stderr)
            sys.exit(1)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(args.input, **read_kwargs)
        df = df.rename(columns={"id": "zinc_id"})[["zinc_id", "smiles"]]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Let pandas infer compression from suffix (e.g., .pkl.gz) or save plain .pkl
    compression = "infer"

    df.to_pickle(args.output, compression=compression)
    print(f"Saved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()

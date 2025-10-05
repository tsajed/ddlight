#!/usr/bin/env python3
"""
cxsmiles_to_pickle.py
Read one or many Enamine .cxsmiles files (tab- or whitespace-delimited) and save a single pickle
with just two columns: 'smiles' and 'zinc_id' (renamed from 'id').
"""

import argparse
import gzip
import os
import sys
import glob
import pandas as pd
import bz2


def infer_sep_from_path(path: str, explicit_sep: str | None) -> str:
    """Infer delimiter per file if not explicitly provided."""
    if explicit_sep is not None:
        return explicit_sep
    # Peek first non-empty line
    if path.endswith(".gz"):
        opener = gzip.open
    elif path.endswith(".bz2"):
        opener = bz2.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        first = fh.readline()
        while first and not first.strip():
            first = fh.readline()
    if not first:
        # Fallback; treat as whitespace
        return r"\s+"
    return "\t" if "\t" in first else r"\s+"


def main():
    ap = argparse.ArgumentParser(
        description="Convert one or many Enamine .cxsmiles to a 2-column pickle (smiles, zinc_id)."
    )
    ap.add_argument("input", help="Path to a .cxsmiles(.gz) file OR a directory containing such files")
    ap.add_argument("-o", "--output", default="smiles_zinc_id.pkl",
                    help="Output pickle path (e.g., smiles_zinc_id.pkl or .pkl.gz). Default: %(default)s")
    ap.add_argument("--sep", default=None,
                    help="Delimiter override for ALL files. If not provided, inferred per file (tab or whitespace).")
    ap.add_argument("--chunksize", type=int, default=None,
                    help="Optional: read in chunks (rows per chunk). If omitted, reads each file in one pass.")
    ap.add_argument("--glob", dest="pattern", default="*.cxsmiles*",
                    help="When INPUT is a directory, glob pattern for files to include. Default: %(default)s")
    args = ap.parse_args()

    # Collect input files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
        if not files:
            print(f"No files matched {args.pattern!r} in {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        files = [args.input]

    usecols = ["smiles", "id"]
    chunks = []

    for fp in files:
        sep = infer_sep_from_path(fp, args.sep)

        # Choose parser engine (fast path when clearly tab-delimited)
        engine = "python"
        if sep == "\t":
            try:
                import pyarrow  # noqa
                engine = "pyarrow"
            except Exception:
                engine = "c"

        read_kwargs = dict(
            sep=sep,
            engine=engine,
            usecols=usecols,
            dtype={"smiles": "string", "id": "string"},
            na_filter=False,
        )

        if args.chunksize:
            for chunk in pd.read_csv(fp, chunksize=args.chunksize, **read_kwargs):
                chunk = chunk.rename(columns={"id": "zinc_id"})[["zinc_id", "smiles"]]
                chunks.append(chunk)
        else:
            df = pd.read_csv(fp, **read_kwargs)
            df = df.rename(columns={"id": "zinc_id"})[["zinc_id", "smiles"]]
            chunks.append(df)

    if not chunks:
        print("No data read; check input files/columns.", file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(chunks, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Let pandas infer compression from suffix (e.g., .pkl.gz) or save plain .pkl
    df_all.to_pickle(args.output, compression="infer")
    print(f"Saved {len(df_all):,} rows from {len(files)} file(s) to {args.output}")


if __name__ == "__main__":
    main()

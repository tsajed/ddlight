#!/usr/bin/env python3
"""
cxsmiles_to_parquet.py
Read one or many Enamine .cxsmiles files (supports .bz2/.gz) and save ONE Parquet file
with just two columns: 'smiles' and 'zinc_id' (renamed from 'id'), streaming in batches.
"""

import argparse
import os
import sys
import glob

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import gzip, bz2, lzma  # only used for lightweight sep inference

# ---- progress support (minimal + optional) ----
try:
    from tqdm import tqdm  # if unavailable, we fall back to occasional prints
except Exception:
    tqdm = None
# ------------------------------------------------

def infer_sep_from_path(path: str, explicit_sep: str | None) -> str:
    """Infer delimiter per file if not explicitly provided (tabs preferred, else whitespace)."""
    if explicit_sep is not None:
        return explicit_sep
    p = path.lower()
    if p.endswith(".gz"):
        opener = gzip.open
    elif p.endswith(".bz2"):
        opener = bz2.open
    elif p.endswith(".xz"):
        opener = lzma.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        first = fh.readline()
        while first and not first.strip():
            first = fh.readline()
        if first.startswith("\ufeff"):
            first = first.lstrip("\ufeff")
    if not first:
        return r"\s+"
    return "\t" if "\t" in first else r"\s+"


def main():
    ap = argparse.ArgumentParser(
        description="Convert one or many Enamine .cxsmiles(.bz2/.gz) to ONE Parquet (smiles, zinc_id)."
    )
    ap.add_argument("input", help="Path to a .cxsmiles(.bz2/.gz) file OR a directory containing such files")
    ap.add_argument(
        "-o", "--output",
        default="smiles_zinc_id.parquet",
        help="Output Parquet path. Default: %(default)s"
    )
    ap.add_argument(
        "--sep", default=None,
        help="Delimiter override for ALL files. If not provided, inferred per file (tab or whitespace)."
    )
    ap.add_argument(
        "--glob", dest="pattern", default="*.cxsmiles*",
        help="When INPUT is a directory, glob pattern for files to include. Default: %(default)s"
    )
    ap.add_argument(
        "--batch-rows", type=int, default=None,
        help="Optional: target rows per written batch/row group (default: use file/batch sizes as-is)."
    )
    ap.add_argument(
        "--progress", action="store_true",
        help="Show progress (files processed, rows written)."
    )
    args = ap.parse_args()

    # Gather input files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
        if not files:
            print(f"No files matched {args.pattern!r} in {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        files = [args.input]

    # Ensure output dir exists
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    if not os.path.isdir(out_dir):
        print(f"Output directory does not exist: {out_dir}", file=sys.stderr)
        sys.exit(1)

    writer = None
    total_rows = 0
    wanted_cols = ("smiles", "id")

    # Wrap file iterator with tqdm if requested/available
    file_iter = files
    if args.progress and tqdm is not None:
        file_iter = tqdm(files, desc="files", unit="file")

    for fp in file_iter:
        sep = infer_sep_from_path(fp, args.sep)

        # Configure Arrow CSV streaming reader
        parse_opts = pacsv.ParseOptions(delimiter="\t" if sep == "\t" else " ")
        read_opts = pacsv.ReadOptions(block_size=1 << 26)  # 64MB blocks; tweak if needed

        conv_opts = pacsv.ConvertOptions(
            include_columns=list(wanted_cols),   # only read these two columns
            include_missing_columns=False,       # error if either is missing
            strings_can_be_null=True             # safe for empty cells in string cols
        )

        reader = pacsv.open_csv(
            fp,
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=conv_opts,           # <-- add this
        )

        # Lazily create ParquetWriter with our 2-column schema
        if writer is None:
            schema = pa.schema([("smiles", pa.string()), ("zinc_id", pa.string())])
            writer = pq.ParquetWriter(args.output, schema, compression="zstd")

        # Per-file row counter + optional tqdm bar for rows
        rows_this_file = 0
        row_pbar = None
        if args.progress and tqdm is not None:
            row_pbar = tqdm(desc=os.path.basename(fp), unit="rows", leave=False)

        batch_accum = []
        accum_rows = 0
        for batch in reader:
            cols = batch.schema.names
            if not all(c in cols for c in wanted_cols):
                continue

            b2 = batch.select(wanted_cols)
            tbl = pa.Table.from_batches([b2])
            out_tbl = pa.table({"smiles": tbl.column("smiles"),
                                "zinc_id": tbl.column("id")})

            n = out_tbl.num_rows
            rows_this_file += n
            total_rows += n

            if args.batch_rows:
                batch_accum.append(out_tbl)
                accum_rows += n
                if accum_rows >= args.batch_rows:
                    writer.write_table(pa.concat_tables(batch_accum))
                    batch_accum, accum_rows = [], 0
            else:
                writer.write_table(out_tbl)

            if row_pbar is not None:
                row_pbar.update(n)
            elif args.progress and tqdm is None and (total_rows % 1_000_000 == 0):
                # fallback prints every ~1M rows if tqdm isn't installed
                print(f"[progress] {total_rows:,} rows written...", file=sys.stderr)

        if args.batch_rows and batch_accum:
            writer.write_table(pa.concat_tables(batch_accum))

        if row_pbar is not None:
            row_pbar.close()
            # also echo a one-line summary per file
            tqdm.write(f"[progress] {os.path.basename(fp)}: {rows_this_file:,} rows (total {total_rows:,})")
        elif args.progress and tqdm is None:
            print(f"[progress] {os.path.basename(fp)}: {rows_this_file:,} rows (total {total_rows:,})",
                  file=sys.stderr)

    if writer is not None:
        writer.close()

    print(f"Saved {total_rows:,} rows from {len(files)} file(s) to {args.output}")


if __name__ == "__main__":
    main()

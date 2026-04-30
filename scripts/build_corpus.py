from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end corpus build pipeline.")
    parser.add_argument("--corpus-id", required=True, help="Target corpus identifier.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw docs directory.")
    parser.add_argument("--docs-path", default="data/processed/docs.json", help="Normalized docs output.")
    parser.add_argument("--chunks-path", default="data/processed/chunks.json", help="Chunks output.")
    parser.add_argument("--index-root", default="data/index", help="Index output directory.")
    parser.add_argument("--source-prefix", default="/docs", help="Prefix for source URLs.")
    parser.add_argument("--chunk-size-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=70)
    return parser.parse_args()


def _run_step(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    ingest_cmd = [
        python_exec,
        str(project_root / "scripts" / "ingest.py"),
        "--raw-dir",
        args.raw_dir,
        "--output-path",
        args.docs_path,
        "--source-prefix",
        args.source_prefix,
    ]
    chunk_cmd = [
        python_exec,
        str(project_root / "scripts" / "chunk.py"),
        "--docs-path",
        args.docs_path,
        "--output-path",
        args.chunks_path,
        "--chunk-size-words",
        str(args.chunk_size_words),
        "--overlap-words",
        str(args.overlap_words),
    ]
    index_cmd = [
        python_exec,
        str(project_root / "scripts" / "build_index.py"),
        "--corpus-id",
        args.corpus_id,
        "--chunks-path",
        args.chunks_path,
        "--index-root",
        args.index_root,
    ]

    _run_step(ingest_cmd)
    _run_step(chunk_cmd)
    _run_step(index_cmd)
    print(f"Corpus build complete for '{args.corpus_id}'.")


if __name__ == "__main__":
    main()

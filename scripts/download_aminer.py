#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URLS = {
    "dblp_v12": "https://originalstatic.aminer.cn/misc/dblp.v12.7z",
    "dblp_v13": "https://originalstatic.aminer.cn/misc/dblp.v13.7z",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(URLS))
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--sha256", help="Optional publisher/user-supplied SHA-256")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(URLS[args.dataset]).name
    urllib.request.urlretrieve(URLS[args.dataset], destination)
    hasher = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if args.sha256 and digest.casefold() != args.sha256.casefold():
        destination.unlink()
        raise SystemExit(f"Checksum mismatch: expected {args.sha256}, observed {digest}")
    print(f"Downloaded {destination} (sha256={digest})")
    print("Extract with: 7z x " + str(destination) + " -o" + str(output_dir / args.dataset))


if __name__ == "__main__":
    main()

"""Build a deterministic source zip without caches, datasets, or weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination zip path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directories = {"src", "examples", "tests", "scripts", "configs", "docs"}
    root_files = {
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
    }
    excluded_parts = {"__pycache__", ".pytest_cache", "build", "dist"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.resolve() == output:
            continue
        relative = path.relative_to(root)
        if any(
            part in excluded_parts or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.parts[0] not in directories and str(relative) not in root_files:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append((path, relative))
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo("AttestRF/" + relative.as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    print(f"Built {output.name}: {len(files)} files")


if __name__ == "__main__":
    main()

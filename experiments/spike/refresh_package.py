#!/usr/bin/env python3
"""Export both engines from an external Spike tree without changing its index."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


SHARED = (
    "riscv/encoding.h",
    "riscv/processor.cc",
    "riscv/processor.h",
    "riscv/riscv.mk.in",
    "riscv/trap.h",
)
ADDED = (
    "riscv/insns/xtile_cfg.h",
    "riscv/insns/xtile_run.h",
    "riscv/insns/xtile_set_ab.h",
    "riscv/insns/xtile_set_c.h",
    "riscv/insns/xtile_set_zp.h",
    "riscv/insns/xrequant_cfg.h",
    "riscv/insns/xrequant_run.h",
    "riscv/insns/xrequant_set_io.h",
    "riscv/insns/xrequant_set_mul.h",
    "riscv/insns/xrequant_set_zp.h",
)
FILES = SHARED + ADDED


def git(directory, *arguments):
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="External modified Spike Git checkout")
    args = parser.parse_args()
    source = args.source.resolve()
    package = Path(__file__).resolve().parent
    base = (package / "base_commit.txt").read_text().strip()
    if not base or len(base) != 40 or any(c not in "0123456789abcdef" for c in base):
        raise RuntimeError("base_commit.txt must contain one full Git commit ID")
    git(source, "cat-file", "-e", f"{base}^{{commit}}")
    snapshot = {}
    for name in FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unsupported source file: {path}")
        snapshot[name] = path.read_bytes()

    with tempfile.TemporaryDirectory(prefix="spike-package-export-") as temp:
        temp_root = Path(temp)
        original = temp_root / "original"
        working = temp_root / "working"
        for name in SHARED:
            target = original / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(source, "show", f"{base}:{name}"))
        for name in ADDED:
            exists = subprocess.run(
                ["git", "-C", str(source), "cat-file", "-e", f"{base}:{name}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            if exists:
                raise RuntimeError(f"Expected a new file absent from the base: {name}")

        shutil.copytree(original, working)
        git(working, "init", "-q")
        git(working, "add", "--", *SHARED)
        for name, data in snapshot.items():
            target = working / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        git(working, "add", "-N", "--", *ADDED)
        patch = git(working, "diff", "--binary", "--full-index", "--", *FILES)
        if patch.count(b"new file mode") != len(ADDED):
            raise RuntimeError("Patch does not contain all ten new instruction headers")
        patch_path = temp_root / "xtile_xrequant.patch"
        patch_path.write_bytes(patch)

        git(original, "init", "-q")
        git(original, "apply", "--check", str(patch_path))
        git(original, "apply", "--whitespace=error", str(patch_path))
        for name, data in snapshot.items():
            if (original / name).read_bytes() != data:
                raise RuntimeError(f"Patch/source mismatch: {name}")

        # Update package files only after patch validation has succeeded.
        for name, data in snapshot.items():
            target = package / "modified" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (package / "xtile_xrequant.patch").write_bytes(patch)
        hashes = []
        for name in sorted(FILES):
            digest = hashlib.sha256(snapshot[name]).hexdigest()
            hashes.append(f"{digest}  modified/{name}\n")
        (package / "sources.sha256").write_text("".join(hashes))

    print(f"PASS: exported {len(FILES)} files, including all ten instruction headers.")
    print("PASS: patch applies to the original file subset and reproduces the snapshot.")
    print("Source checkout, Git index, and installed simulator were not changed.")
    print("A clean full Spike build and regression run remain separate checks.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")

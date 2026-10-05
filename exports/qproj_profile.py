#!/usr/bin/env python3
"""Profile named function ranges in a trap-free Spike -l trace.

Example:
  python3 qproj_profile.py --elf build/qproj/qproj_test.elf \
    --trace build/qproj/qproj_trace.log --macs 524288 --outputs 4096 \
    --json build/qproj/b0_profile.json

Requires RISC-V GNU nm, objdump, readelf, and size on PATH. No ML packages.
Addresses come from the ELF, not a hard-coded address range. Only instructions
directly inside selected functions are counted; add out-of-line callees using
additional --function options. Default function: qproj_accumulate.
--region NAME=START:END optionally measures a known PC range (end exclusive).
For the current scalar q_proj, the innermost backward-branch loop containing
mulw is detected from disassembly. Its executed instructions and scalar MAC
multiplies are reported. A unique mul outside this loop identifies the current
requantization multiplication. --inner-loop and --requant-pc can override these
choices after code changes. Ambiguous layouts report unavailable measurements
instead of guessing. --macs and --outputs remain expected workload counts.
Use a freshly generated trace from the exact ELF being profiled.
These counts are ISA-level measurements, not processor cycles or cache misses.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


TRACE_PATTERN = re.compile(
    r"core\s+(\d+):\s+(?:[0-3]\s+)?"
    r"(0x[0-9a-fA-F]+)\s+\(0x([0-9a-fA-F]+)\)"
)
DISASM_PATTERN = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{4,8})\s+(\S+)(?:\s+(.*))?$"
)
BRANCH_TARGET_PATTERN = re.compile(
    r"(?:^|[,\s])(?:0x)?([0-9a-fA-F]+)(?:\s+<[^>]*>)?\s*$"
)
LOADS = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "lwsp", "ldsp"}
STORES = {"sb", "sh", "sw", "sd", "swsp", "sdsp"}
MULTIPLIES = {"mul", "mulw", "mulh", "mulhu", "mulhsu"}
CONTROL = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
    "j", "jr", "jal", "jalr", "ret", "call", "tail",
}
CONDITIONAL_BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}
ALU = {
    "add", "addi", "addw", "addiw", "addi4spn", "addi16sp", "sub", "subw",
    "and", "andi", "or", "ori", "xor", "xori", "sll", "slli", "sllw", "slliw",
    "srl", "srli", "srlw", "srliw", "sra", "srai", "sraw", "sraiw",
    "slt", "slti", "sltu", "sltiu", "lui", "auipc", "li", "mv", "nop",
    "sext.w", "neg", "negw", "not", "seqz", "snez",
}


def run_tool(prefix, name, *args):
    try:
        return subprocess.check_output([prefix + name, *args], text=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Tool not found: {prefix + name}") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Tool failed: {prefix + name}") from exc


def category(mnemonic):
    op = mnemonic.removeprefix("c.")
    if op in LOADS:
        return "load"
    if op in STORES:
        return "store"
    if op in MULTIPLIES:
        return "multiply"
    if op in CONTROL:
        return "branch/jump"
    if op in ALU:
        return "integer ALU"
    if op in {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}:
        return "divide/remainder"
    if op.startswith("v") or (op.startswith("f") and not op.startswith("fence")):
        return "floating point/vector"
    return "other"


def read_symbols(output, names):
    symbols = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[3] in names and fields[2].lower() == "t":
            start, length = int(fields[0], 16), int(fields[1], 16)
            if length > 0:
                symbols[fields[3]] = (start, start + length)
    missing = set(names) - symbols.keys()
    if missing:
        raise ValueError("Missing sized text symbols: " + ", ".join(sorted(missing)))
    return symbols


def read_disassembly(output):
    instructions = {}
    for line in output.splitlines():
        match = DISASM_PATTERN.match(line)
        if match:
            pc = int(match[1], 16)
            encoding = int(match[2], 16)
            operands = (match[4] or "").split("#", 1)[0].strip()
            instructions[pc] = (encoding, match[3], operands, len(match[2]) // 2)
    return instructions


def in_ranges(pc, ranges):
    return any(start <= pc < end for start, end in ranges)


def count_trace(trace, ranges, instructions, hart=0):
    pcs = Counter()
    matched = 0
    previous_selected = False
    for line in trace:
        match = TRACE_PATTERN.search(line)
        if not match:
            if previous_selected and ("repeat" in line.lower() or "exception" in line.lower()):
                raise ValueError("Repeated or trapping kernel trace needs manual inspection.")
            continue
        if int(match[1]) != hart:
            continue
        matched += 1
        pc = int(match[2], 16)
        previous_selected = in_ranges(pc, ranges)
        if not previous_selected:
            continue
        if pc not in instructions:
            raise ValueError(f"Trace PC {pc:#x} is not an ELF instruction.")
        if int(match[3], 16) != instructions[pc][0]:
            raise ValueError(f"Trace/ELF instruction mismatch at {pc:#x}; regenerate the trace.")
        pcs[pc] += 1
    if not pcs:
        raise ValueError("No selected instructions found. Check the trace, ELF, and function names.")
    return pcs, matched


def parse_region(value):
    try:
        name, bounds = value.split("=", 1)
        start, end = (int(x, 0) for x in bounds.split(":"))
        if not name or start >= end:
            raise ValueError()
        return name, start, end
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use NAME=START:END, with an exclusive end.") from exc


def parse_bounds(value):
    _, start, end = parse_region("inner_loop=" + value)
    return start, end


def scalar_loop_summary(pcs, instructions, ranges, expected_macs, expected_outputs,
                        explicit_loop=None, explicit_requant_pc=None):
    """Locate a scalar dot-product loop; never infer it solely from frequency."""
    candidates = []
    if explicit_loop is not None:
        start, end = explicit_loop
        if not any(a <= start < end <= b for a, b in ranges):
            raise ValueError("The explicit inner loop must be inside a selected function.")
        candidates = [(start, end)]
    else:
        for pc, (_, mnemonic, operands, length) in instructions.items():
            if pc not in pcs or mnemonic.removeprefix("c.") not in CONDITIONAL_BRANCHES:
                continue
            target = BRANCH_TARGET_PATTERN.search(operands)
            if not target:
                continue
            start, end = int(target[1], 16), pc + length
            if start >= pc or not any(a <= start < end <= b for a, b in ranges):
                continue
            body = [(addr, item) for addr, item in instructions.items() if start <= addr < end]
            has_mac = any(item[1].removeprefix("c.") == "mulw" for _, item in body)
            other_control = any(addr != pc and category(item[1]) == "branch/jump" for addr, item in body)
            if has_mac and not other_control:
                candidates.append((start, end))
    summary = {
        "inner_loop": None,
        "inner_loop_instructions": None,
        "outside_inner_loop_instructions": None,
        "inner_loop_percent": None,
        "observed_scalar_macs": None,
        "scalar_macs_match_expected": None,
        "requantization_pc": None,
        "requantizations": None,
        "requantizations_match_expected_outputs": None,
        "detection_notes": [],
    }
    if len(candidates) == 1:
        start, end = candidates[0]
        inner = sum(n for pc, n in pcs.items() if start <= pc < end)
        mac_pcs = [pc for pc in instructions if start <= pc < end and instructions[pc][1].removeprefix("c.") == "mulw"]
        if not inner or not mac_pcs:
            raise ValueError("Inner loop has no traced instructions or scalar mulw operations.")
        observed_macs = sum(pcs[pc] for pc in mac_pcs)
        total = sum(pcs.values())
        summary.update({
            "inner_loop": {"start": hex(start), "end_exclusive": hex(end), "mac_instruction_pcs": [hex(pc) for pc in mac_pcs]},
            "inner_loop_instructions": inner,
            "outside_inner_loop_instructions": total - inner,
            "inner_loop_percent": 100 * inner / total,
            "observed_scalar_macs": observed_macs,
            "scalar_macs_match_expected": observed_macs == expected_macs,
        })
        req_candidates = [pc for pc in pcs if not start <= pc < end and instructions[pc][1] == "mul"]
    else:
        req_candidates = []
        summary["detection_notes"].append("Scalar inner loop is missing or ambiguous; specify --inner-loop START:END.")
    if explicit_requant_pc is not None:
        if explicit_requant_pc not in pcs or category(instructions[explicit_requant_pc][1]) != "multiply":
            raise ValueError("--requant-pc must identify an executed multiply inside a selected function.")
        req_candidates = [explicit_requant_pc]
    if len(req_candidates) == 1:
        req_pc = req_candidates[0]
        summary.update({
            "requantization_pc": hex(req_pc),
            "requantizations": pcs[req_pc],
            "requantizations_match_expected_outputs": pcs[req_pc] == expected_outputs,
        })
    else:
        summary["detection_notes"].append("Requantization multiply is missing or ambiguous; specify --requant-pc ADDRESS.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--function", action="append", help="Repeat to include out-of-line callees.")
    parser.add_argument("--macs", type=int, required=True, help="Known workload MAC count, not inferred from mul.")
    parser.add_argument("--outputs", type=int, required=True)
    parser.add_argument("--prefix", default="riscv64-unknown-elf-")
    parser.add_argument("--hart", type=int, default=0)
    parser.add_argument("--region", action="append", type=parse_region, default=[])
    parser.add_argument("--inner-loop", type=parse_bounds, help="Override scalar loop detection: START:END, end exclusive.")
    parser.add_argument("--requant-pc", type=lambda value: int(value, 0), help="Override the requantization multiply PC, e.g. 0x80000190.")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.macs <= 0 or args.outputs <= 0:
        parser.error("MAC and output counts must be positive.")

    names = args.function or ["qproj_accumulate"]
    symbols = read_symbols(run_tool(args.prefix, "nm", "-S", "--defined-only", str(args.elf)), names)
    instructions = read_disassembly(run_tool(args.prefix, "objdump", "-d", "-M", "no-aliases", str(args.elf)))
    with args.trace.open() as trace:
        pcs, matched = count_trace(trace, list(symbols.values()), instructions, args.hart)
    mix, histogram = Counter(), Counter()
    for pc, count in pcs.items():
        op = instructions[pc][1]
        mix[category(op)] += count
        histogram[op] += count
    total = sum(pcs.values())
    loop_summary = scalar_loop_summary(pcs, instructions, list(symbols.values()), args.macs,
                                      args.outputs, args.inner_loop, args.requant_pc)
    size_output = run_tool(args.prefix, "size", str(args.elf))
    size_line = next((line.split() for line in size_output.splitlines() if re.match(r"^\s*\d+\s+\d+\s+\d+", line)), None)
    attributes = run_tool(args.prefix, "readelf", "-A", str(args.elf))
    arch = re.search(r'Tag_RISCV_arch:\s*"([^"]+)"', attributes)
    report = {
        "elf": str(args.elf.resolve()),
        "elf_sha256": hashlib.sha256(args.elf.read_bytes()).hexdigest(),
        "trace": str(args.trace.resolve()),
        "isa_attribute": arch[1] if arch else None,
        "functions": {name: {"start": hex(start), "end_exclusive": hex(end), "symbol_bytes": end-start}
                      for name, (start, end) in symbols.items()},
        "kernel_instructions": total,
        "instruction_mix": dict(mix),
        "mnemonic_counts": dict(histogram),
        "macs": args.macs,
        "output_elements": args.outputs,
        "macs_per_instruction": args.macs / total,
        "instructions_per_mac": total / args.macs,
        "instructions_per_output": total / args.outputs,
        "loads_per_mac": mix["load"] / args.macs,
        "stores_per_mac": mix["store"] / args.macs,
        "memory_instructions": mix["load"] + mix["store"],
        "memory_instructions_per_mac": (mix["load"] + mix["store"]) / args.macs,
        "elf_size_bytes": {"text_including_read_only_data": int(size_line[0]), "data": int(size_line[1]), "bss": int(size_line[2])} if size_line else None,
        "regions": {name: {"start": hex(start), "end_exclusive": hex(end),
                            "instructions": sum(n for pc, n in pcs.items() if start <= pc < end)}
                    for name, start, end in args.region},
        "matched_trace_records": matched,
        "scope": "Instructions directly inside selected function symbols; no cycle or cache measurements.",
        **loop_summary,
    }
    print("ELF ISA:", report["isa_attribute"])
    for name, details in report["functions"].items():
        print(f"Function {name}: {details['start']}..{details['end_exclusive']} (exclusive), {details['symbol_bytes']} bytes")
    print(f"Kernel instructions: {total:,}")
    if report["inner_loop"]:
        loop = report["inner_loop"]
        print(f"Inner loop: {loop['start']}..{loop['end_exclusive']} (exclusive)")
        print(f"Inner-loop instructions: {report['inner_loop_instructions']:,}")
        print(f"Outside-inner-loop instructions: {report['outside_inner_loop_instructions']:,}")
        print(f"Inner-loop share: {report['inner_loop_percent']:.3f}%")
    print(f"MACs (expected workload): {args.macs:,}")
    if report["observed_scalar_macs"] is not None:
        status = "MATCH" if report["scalar_macs_match_expected"] else "MISMATCH"
        print(f"MACs (observed scalar mulw executions): {report['observed_scalar_macs']:,} [{status}]")
    print(f"Output elements (expected workload): {args.outputs:,}")
    if report["requantizations"] is not None:
        status = "MATCH" if report["requantizations_match_expected_outputs"] else "MISMATCH"
        print(f"Requantizations: {report['requantizations']:,} [{status}] at {report['requantization_pc']}")
    for note in report["detection_notes"]:
        print("Detection note:", note)
    print("Instruction mix:")
    for name, count in sorted(mix.items()):
        print(f"  {name:24s} {count:12,d}  {100*count/total:7.3f}%")
    for key in ["macs_per_instruction", "instructions_per_mac", "instructions_per_output", "loads_per_mac", "stores_per_mac", "memory_instructions_per_mac"]:
        print(f"{key}: {report[key]:.9f}")
    print("ELF sizes (text includes read-only data):", report["elf_size_bytes"])
    for name, details in report["regions"].items():
        print(f"Region {name}: {details['instructions']:,} instructions")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print("Saved:", args.json)
    if mix["floating point/vector"]:
        raise ValueError("Floating-point/vector operations were found in the selected functions.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"Profile error: {exc}", file=sys.stderr)
        sys.exit(1)

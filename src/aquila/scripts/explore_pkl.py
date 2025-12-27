#!/usr/bin/env python3
"""
Quick PKL (pickle) structure explorer

Usage:
  python explore_pkl.py data.pkl [--max-depth N] [--max-items N] [--show-values]

Notes:
  - WARNING: Unpickling arbitrary files is unsafe. Only inspect trusted .pkl files.
"""

import argparse
import os
import sys
import pickle
from collections import deque
from dataclasses import is_dataclass, asdict

try:
    import numpy as np
except Exception:
    np = None


def _shorten_text(s: str, max_bytes: int) -> str:
    if s is None:
        return "None"
    if len(s) <= max_bytes:
        return s
    return s[: max(0, max_bytes - 3)] + "..."


def _safe_type_name(x) -> str:
    t = type(x)
    if getattr(t, "__module__", None) and getattr(t, "__name__", None):
        return f"{t.__module__}.{t.__name__}"
    return str(t)


def _is_numpy_array(x) -> bool:
    return np is not None and isinstance(x, np.ndarray)


def _format_leaf(x, max_bytes: int):
    """Return a compact, single-line summary for leaf-ish values."""
    if x is None or isinstance(x, (bool, int, float)):
        return str(x)

    if isinstance(x, str):
        return f"str(len={len(x)}): {_shorten_text(x, max_bytes)}"

    if isinstance(x, (bytes, bytearray)):
        return f"bytes(len={len(x)})"

    if _is_numpy_array(x):
        size = int(x.size)
        return f"ndarray(shape={x.shape}, dtype={x.dtype}, size={size})"

    # Common leaf-ish
    if isinstance(x, (complex,)):
        return str(x)

    return f"<{_safe_type_name(x)}>"


def _iter_container(x):
    """Yield (key, value) pairs for containers."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, v
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield i, v
    elif isinstance(x, set):
        # sets are unordered; show stable-ish order by repr
        for i, v in enumerate(sorted(list(x), key=lambda z: repr(z))):
            yield i, v
    else:
        return


def _as_mapping_if_dataclass_or_object(x):
    """Try to turn dataclass / object with __dict__ into a mapping for exploration."""
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            pass

    # objects with __dict__
    if hasattr(x, "__dict__") and isinstance(getattr(x, "__dict__", None), dict):
        # avoid huge / private spam: still show all, but user can control max-items / depth
        return x.__dict__

    return None


def explore(obj, *, max_depth, max_items, max_bytes, show_values):
    stats = {
        "containers": 0,
        "leaves": 0,
        "dicts": 0,
        "lists": 0,
        "tuples": 0,
        "sets": 0,
        "numpy_arrays": 0,
        "numpy_total_elems": 0,
        "objects": 0,
    }

    def is_container(x):
        if isinstance(x, (dict, list, tuple, set)):
            return True
        if _is_numpy_array(x):
            return False
        mapped = _as_mapping_if_dataclass_or_object(x)
        return mapped is not None

    def print_node(prefix, label, x, depth):
        tname = _safe_type_name(x)
        if isinstance(x, dict):
            print(f"{prefix}📁 {label}: dict(len={len(x)})")
        elif isinstance(x, list):
            print(f"{prefix}📁 {label}: list(len={len(x)})")
        elif isinstance(x, tuple):
            print(f"{prefix}📁 {label}: tuple(len={len(x)})")
        elif isinstance(x, set):
            print(f"{prefix}📁 {label}: set(len={len(x)})")
        elif _is_numpy_array(x):
            stats["numpy_arrays"] += 1
            stats["numpy_total_elems"] += int(x.size)
            print(f"{prefix}📄 {label}: ndarray(shape={x.shape}, dtype={x.dtype}, size={int(x.size)})")
            if show_values and x.size <= 20 and x.dtype.kind in ("i", "u", "f", "b"):
                try:
                    print(f"{prefix}   └─ values: {x.ravel()[:20]}")
                except Exception:
                    pass
        else:
            # dataclass/object treated as container later, but still show header
            mapped = _as_mapping_if_dataclass_or_object(x)
            if mapped is not None:
                stats["objects"] += 1
                print(f"{prefix}📁 {label}: object<{tname}> (fields={len(mapped)})")
            else:
                stats["leaves"] += 1
                print(f"{prefix}📄 {label}: {_format_leaf(x, max_bytes)}")

    def inc_container_stats(x):
        stats["containers"] += 1
        if isinstance(x, dict):
            stats["dicts"] += 1
        elif isinstance(x, list):
            stats["lists"] += 1
        elif isinstance(x, tuple):
            stats["tuples"] += 1
        elif isinstance(x, set):
            stats["sets"] += 1

    def recurse(x, prefix="", depth=1, label="root"):
        print_node(prefix, label, x, depth)

        # depth control
        if max_depth != float("inf") and depth >= max_depth:
            if is_container(x):
                print(f"{prefix}  ... (max depth {max_depth} reached)")
            return

        # numpy arrays are leaves here
        if _is_numpy_array(x):
            return

        # if dict/list/tuple/set
        if isinstance(x, (dict, list, tuple, set)):
            inc_container_stats(x)
            it = list(_iter_container(x))
        else:
            mapped = _as_mapping_if_dataclass_or_object(x)
            if mapped is None:
                return
            # treat object mapping like a dict
            inc_container_stats(mapped if isinstance(mapped, dict) else {})
            it = list(mapped.items())

        if len(it) == 0:
            return

        # item limit
        shown = it[:max_items] if max_items > 0 else it
        for k, v in shown:
            k_str = repr(k)
            # make label pretty
            child_label = f"[{k_str}]" if isinstance(x, (dict, set)) else f"[{k}]"
            # for object fields, prefer ".field"
            if not isinstance(x, (dict, list, tuple, set)) and isinstance(k, str):
                child_label = f".{k}"

            # leaf / container decision
            if is_container(v):
                recurse(v, prefix + "  ", depth + 1, child_label)
            else:
                print_node(prefix + "  ", child_label, v, depth + 1)
                if show_values:
                    # show small sequences
                    if isinstance(v, (list, tuple)) and len(v) <= 10:
                        print(f"{prefix}     └─ values: {v}")
                    elif isinstance(v, dict) and len(v) <= 10:
                        try:
                            keys_preview = list(v.keys())
                            print(f"{prefix}     └─ keys: {keys_preview}")
                        except Exception:
                            pass

        if max_items > 0 and len(it) > max_items:
            print(f"{prefix}  ... ({len(it) - max_items} more items not shown; use --max-items)")

    recurse(obj)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Explore PKL (pickle) file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explore_pkl.py data.pkl
  python explore_pkl.py data.pkl --max-depth 2
  python explore_pkl.py data.pkl --max-depth 0          # unlimited
  python explore_pkl.py data.pkl --max-items 20 --show-values
        """,
    )
    parser.add_argument("filename", help="PKL file to explore")
    parser.add_argument("-md", "--max-depth", type=int, default=10,
                        help="Maximum depth to explore (default: 10, 0 = unlimited)")
    parser.add_argument("-mi", "--max-items", type=int, default=30,
                        help="Max items to show per container (default: 30, 0 = show all)")
    parser.add_argument("-mb", "--max-bytes", type=int, default=120,
                        help="Max chars to show for long strings (default: 120)")
    parser.add_argument("--show-values", action="store_true",
                        help="Show small value previews for arrays/sequences")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"❌ Error: file not found: {args.filename}")
        sys.exit(1)

    max_depth = float("inf") if args.max_depth == 0 else args.max_depth

    # SAFETY WARNING
    print("⚠️  Warning: unpickling is unsafe for untrusted files. Only open trusted .pkl files.\n")

    try:
        with open(args.filename, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"❌ Error reading pickle: {e}")
        sys.exit(1)

    print(f"🔍 Exploring PKL: {args.filename}")
    print(f"📏 Max depth: {'unlimited' if args.max_depth == 0 else args.max_depth}")
    print(f"📦 Max items per container: {'all' if args.max_items == 0 else args.max_items}")
    print("=" * 60)

    stats = explore(
        obj,
        max_depth=max_depth,
        max_items=args.max_items,
        max_bytes=args.max_bytes,
        show_values=args.show_values,
    )

    print("=" * 60)
    # Summary
    summary = [
        f"containers={stats['containers']}",
        f"leaves={stats['leaves']}",
        f"dicts={stats['dicts']}",
        f"lists={stats['lists']}",
        f"tuples={stats['tuples']}",
        f"sets={stats['sets']}",
        f"objects={stats['objects']}",
    ]
    if np is not None:
        summary.append(f"numpy_arrays={stats['numpy_arrays']}")
        summary.append(f"numpy_total_elems={stats['numpy_total_elems']}")
    print("📊 Summary: " + ", ".join(summary))
    print("✅ Done!")


if __name__ == "__main__":
    main()
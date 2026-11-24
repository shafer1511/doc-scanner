import ast
import sys
from pathlib import Path

FORBIDDEN_CALLS = {
    "Canny",
    "HoughLines",
    "HoughLinesP",
    "hough_line",
    "probabilistic_hough_line",
    "threshold",
    "adaptiveThreshold",
    "threshold_otsu",
    "threshold_sauvola",
    "threshold_niblack",
    "threshold_local",
    "threshold_yen",
    "threshold_isodata",
    "threshold_li",
    "threshold_mean",
    "threshold_minimum",
    "threshold_triangle",
}

FORBIDDEN_IMPORT_MODULES = {
    "skimage.feature",
    "skimage.transform",
    "skimage.filters",
}


def scan_file(path: Path) -> list[str]:
    """Scan one Python file's AST for forbidden library calls or imports."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORT_MODULES or alias.name.startswith("skimage."):
                    violations.append(f"{path}:{node.lineno} forbidden import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in FORBIDDEN_IMPORT_MODULES or mod.startswith("skimage."):
                violations.append(f"{path}:{node.lineno} forbidden import from '{mod}'")
            for alias in node.names:
                if alias.name in FORBIDDEN_CALLS:
                    violations.append(f"{path}:{node.lineno} forbidden imported symbol '{alias.name}'")
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_CALLS:
                violations.append(f"{path}:{node.lineno} forbidden call '{name}'")

    return violations


def main() -> int:
    core_dir = Path("src/core")
    if not core_dir.exists():
        print("ERROR: src/core directory not found.", file=sys.stderr)
        return 1

    py_files = sorted(core_dir.glob("*.py"))
    if not py_files:
        print("ERROR: No Python files found in src/core.", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for f in py_files:
        all_violations.extend(scan_file(f))

    if all_violations:
        for v in all_violations:
            print(f"VIOLATION: {v}", file=sys.stderr)
        return 1

    print(f"AST Verification PASSED ({len(py_files)} files checked in src/core, 0 violations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

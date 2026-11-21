"""AST verification utility to ensure no forbidden library functions are called in src/core/."""
import ast
import sys
from pathlib import Path

FORBIDDEN_RULES = {
    "canny.py": [
        ("cv2", "Canny"),
        ("skimage.feature", "canny"),
        ("feature", "canny"),
    ],
    "hough.py": [
        ("cv2", "HoughLines"),
        ("cv2", "HoughLinesP"),
        ("cv2", "HoughLinesPointSet"),
        ("skimage.transform", "hough_line"),
        ("skimage.transform", "hough_line_peaks"),
        ("transform", "hough_line"),
        ("transform", "hough_line_peaks"),
    ],
    "thresholding.py": [
        ("cv2", "threshold"),
        ("cv2", "adaptiveThreshold"),
        ("skimage.filters", "threshold_otsu"),
        ("skimage.filters", "threshold_sauvola"),
        ("skimage.filters", "threshold_local"),
        ("filters", "threshold_otsu"),
        ("filters", "threshold_sauvola"),
        ("filters", "threshold_local"),
    ],
}


class ForbiddenCallVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.forbidden_patterns = FORBIDDEN_RULES.get(filename, [])
        self.violations = []

    def visit_Call(self, node):
        # Check func attribute like cv2.Canny
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            val_name = ""
            if isinstance(node.func.value, ast.Name):
                val_name = node.func.value.id
            elif isinstance(node.func.value, ast.Attribute):
                val_name = f"{node.func.value.value.id}.{node.func.value.attr}" if isinstance(node.func.value.value, ast.Name) else ""
            attr_name = node.func.attr
            
            for mod, fn in self.forbidden_patterns:
                if (val_name == mod and attr_name == fn) or (attr_name == fn and mod in val_name):
                    self.violations.append((node.lineno, f"{val_name}.{attr_name}"))
        elif isinstance(node.func, ast.Name):
            for _, fn in self.forbidden_patterns:
                if node.func.id == fn:
                    self.violations.append((node.lineno, node.func.id))
        self.generic_visit(node)


def verify_file(filepath: Path) -> bool:
    content = filepath.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(filepath))
    visitor = ForbiddenCallVisitor(filepath.name)
    visitor.visit(tree)
    if visitor.violations:
        print(f"[AST VIOLATION] in {filepath.name}:")
        for lineno, call in visitor.violations:
            print(f"  Line {lineno}: Forbidden call detected: {call}")
        return False
    print(f"[AST PASSED] {filepath.name} has zero forbidden library calls.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/verify_ast.py <file_path>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"File not found: {target}")
        sys.exit(1)
    passed = verify_file(target)
    sys.exit(0 if passed else 1)

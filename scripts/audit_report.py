import csv
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
from lxml import etree


BLACKLIST_PHRASES = [
    "trong bối cảnh",
    "đóng vai trò then chốt",
    "đóng vai trò quan trọng",
    "không chỉ",
    "mà còn",
    "một cách hiệu quả",
    "một cách toàn diện",
    "một cách mượt mà",
    "tối ưu hóa trải nghiệm",
    "nhìn chung",
    "tóm lại",
    "kết luận lại",
    "có thể thấy rằng",
    "đáng chú ý là",
]

RAW_MARKDOWN_TOKENS = ("**", "$$", "\\frac", "```", "`", "\\Delta", "^*")
ALLOWED_FONTS = {"Times New Roman", "Courier New", "Cambria Math"}


def _word_count(sentence: str) -> int:
    clean = re.sub(r"`[^`]+`|\$[^$]+\$|\*\*|\*", " ", sentence)
    tokens = [t for t in re.split(r"\s+", clean.strip()) if t]
    return len(tokens)


def audit_markdown(md_path: Path) -> list[str]:
    errors: list[str] = []
    if not md_path.exists():
        return [f"Missing markdown report: {md_path}"]

    text = md_path.read_text(encoding="utf-8")
    lower = text.lower()

    for phrase in BLACKLIST_PHRASES:
        if phrase in lower:
            errors.append(f"MD blacklist phrase detected: '{phrase}'")

    ai_bullets = re.findall(r"^\s*-\s+\*\*[^*]+:\*\*", text, flags=re.MULTILINE)
    if ai_bullets:
        errors.append(f"MD AI-style '- **Title:**' bullets found: {len(ai_bullets)}")

    in_code = False
    in_cover = False
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "<!-- COVER_START -->":
            in_cover = True
            continue
        if line == "<!-- COVER_END -->":
            in_cover = False
            continue
        if line.startswith("<!--"):
            continue
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or in_cover or not line:
            continue
        if (
            line.startswith("#")
            or line.startswith("|")
            or line.startswith("$$")
            or line.startswith("![")
            or (line.startswith("*") and line.endswith("*"))
            or re.match(r"^\d+\.\s", line)
        ):
            continue
        check_line = line[2:].strip() if line.startswith("- ") else line
        sentences = re.split(r"(?<!\d)\.(?!\d)\s+", check_line)
        for s in sentences:
            wc = _word_count(s)
            if wc > 25:
                errors.append(f"MD line {line_no} sentence > 25 words ({wc}): '{s[:85]}...'")

    for cid in ("dev_01", "dev_07", "dev_03", "dev_29", "dev_05", "dev_32"):
        if cid not in text:
            errors.append(f"MD missing required case study ID: {cid}")
    for lbl in (
        "*Đầu vào & Điều kiện chụp:*",
        "*Điểm gãy trong pipeline:*",
        "*Bằng chứng trên ảnh trung gian:*",
        "*Nguyên nhân & Tham số liên quan:*",
    ):
        cnt = text.count(lbl)
        if cnt < 6:
            errors.append(f"MD bullet label '{lbl}' appears {cnt} times (< 6)")

    dev_rows = list(csv.DictReader(open("outputs/results_development.csv", encoding="utf-8")))
    test_rows = list(csv.DictReader(open("outputs/results_test.csv", encoding="utf-8")))
    syn_rows = list(csv.DictReader(open("outputs/results_synthetic.csv", encoding="utf-8")))
    ab_hough = list(csv.DictReader(open("outputs/ablation_no_hough_suppression.csv", encoding="utf-8")))
    ab_edge = list(csv.DictReader(open("outputs/ablation_no_edge_support.csv", encoding="utf-8")))

    for label, rows in (("development", dev_rows), ("test", test_rows)):
        na_pos = [
            r["image_id"]
            for r in rows
            if r["condition"] != "khong_co_tai_lieu_hop_le" and r.get("iou", "N/A") == "N/A"
        ]
        if na_pos:
            errors.append(f"CSV {label} has positive rows with iou == N/A: {na_pos[:5]}")

    def _pos_str(rows):
        pos = [r for r in rows if r["condition"] != "khong_co_tai_lieu_hop_le"]
        return f"{sum(r['status']=='accepted' for r in pos)}/{len(pos)}"

    expected_tokens = [
        _pos_str(dev_rows),
        _pos_str(test_rows),
        _pos_str(ab_hough),
        _pos_str(ab_edge),
    ]
    for rows in (dev_rows, test_rows):
        pos = [r for r in rows if r["condition"] != "khong_co_tai_lieu_hop_le"]
        ious = [float(r["iou"]) for r in pos if r.get("iou", "N/A") != "N/A"]
        succ_08 = sum(float(r["iou"]) >= 0.8 for r in pos if r.get("iou", "N/A") != "N/A")
        ces = [float(r["corner_error_norm"]) for r in pos if r.get("corner_error_norm", "N/A") != "N/A"]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_ce = float(np.mean(ces)) if ces else 0.0
        expected_tokens.extend([f"{mean_iou:.4f}", f"{succ_08}/{len(pos)}", f"{mean_ce:.4f}"])

    for sp in ("development", "test"):
        for m in ("otsu", "sauvola"):
            mf1 = float(np.mean([float(r["f1"]) for r in syn_rows if r["split"] == sp and r["method"] == m]))
            expected_tokens.append(f"{mf1:.4f}")

    for tok in expected_tokens:
        if tok not in text:
            errors.append(f"MD traceability mismatch: expected CSV value '{tok}' not found")

    return errors


def audit_docx(docx_path: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    if not docx_path.exists():
        return [f"Missing DOCX report: {docx_path}"], 0

    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    m_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    media_count = 0

    with zipfile.ZipFile(docx_path, "r") as zf:
        names = zf.namelist()
        if "word/document.xml" not in names:
            return ["Invalid DOCX: missing word/document.xml"], 0

        doc_xml = etree.fromstring(zf.read("word/document.xml"))
        all_texts = [t.text for t in doc_xml.xpath(".//w:t | .//m:t", namespaces={"w": w_ns, "m": m_ns}) if t.text]
        full_text = " ".join(all_texts)
        full_lower = full_text.lower()

        for phrase in BLACKLIST_PHRASES:
            if phrase in full_lower:
                errors.append(f"DOCX blacklist phrase detected: '{phrase}'")

        for tok in RAW_MARKDOWN_TOKENS:
            if tok in full_text:
                errors.append(f"DOCX raw Markdown/LaTeX token leaked: '{tok}'")
        if re.search(r"(^|\s)#+\s", full_text):
            errors.append("DOCX raw Markdown heading '# ' leaked")

        # Strict OOXML checks: no nested m:oMathPara, valid child ordering
        nested_omath = doc_xml.xpath(".//m:oMathPara//m:oMathPara", namespaces={"m": m_ns})
        if nested_omath:
            errors.append(f"DOCX OOXML corruption: {len(nested_omath)} nested <m:oMathPara> found")

        for ppr in doc_xml.xpath(".//w:pPr", namespaces={"w": w_ns}):
            tags = [etree.QName(c).localname for c in ppr]
            if "shd" in tags:
                s_idx = tags.index("shd")
                for after_tag in ("spacing", "jc"):
                    if after_tag in tags and s_idx > tags.index(after_tag):
                        errors.append(f"DOCX OOXML order error in w:pPr: w:shd appears after w:{after_tag}")

        for mr in doc_xml.xpath(".//m:r", namespaces={"m": m_ns}):
            tags = [f"{etree.QName(c).namespace.split('/')[-1]}:{etree.QName(c).localname}" for c in mr]
            if "math:rPr" in tags and "main:rPr" in tags:
                if tags.index("main:rPr") < tags.index("math:rPr"):
                    errors.append("DOCX OOXML order error in m:r: w:rPr appears before m:rPr")
            if "main:rPr" in tags and "math:t" in tags:
                if tags.index("main:rPr") > tags.index("math:t"):
                    errors.append("DOCX OOXML order error in m:r: w:rPr appears after m:t")

        omaths = doc_xml.xpath(".//m:oMath", namespaces={"m": m_ns})
        if len(omaths) < 6:
            errors.append(f"DOCX has only {len(omaths)} <m:oMath> elements (expected >= 6)")
        ofracs = doc_xml.xpath(".//m:f", namespaces={"m": m_ns})
        if len(ofracs) < 2:
            errors.append(f"DOCX has only {len(ofracs)} stacked fractions <m:f> (expected >= 2)")

        # Font whitelist across all XML parts (document, styles, headers, footers)
        found_fonts = set()
        xml_parts = [n for n in names if n in ("word/document.xml", "word/styles.xml") or n.startswith("word/header") or n.startswith("word/footer")]
        for xp in xml_parts:
            root = etree.fromstring(zf.read(xp))
            for rf in root.xpath(".//w:rFonts", namespaces={"w": w_ns}):
                for _, val in rf.attrib.items():
                    if val:
                        found_fonts.add(val)

        disallowed = found_fonts - ALLOWED_FONTS
        if disallowed:
            errors.append(f"DOCX disallowed fonts detected: {sorted(disallowed)}")

        media_count = len([n for n in names if n.startswith("word/media/")])
        if media_count < 20:
            errors.append(f"DOCX embedded media count = {media_count} (< 20)")

    return errors, media_count


def audit_word_com(docx_path: Path, pdf_path: Path) -> tuple[list[str], int]:
    """Open .docx in real Microsoft Word COM without repair dialog and export PDF."""
    abs_docx = str(docx_path.resolve())
    abs_pdf = str(pdf_path.resolve())
    ps_script = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {{
    # OpenNoRepairDialog=$true (arg 16), ConfirmConversions=$false, ReadOnly=$true
    $doc = $word.Documents.Open('{abs_docx}', $false, $true, $false, '', '', $false, '', '', 0, 0, $false, $false, 0, $true)
    $pages = $doc.ComputeStatistics(2)
    $doc.SaveAs('{abs_pdf}', 17)
    Write-Output "WORD_COM_OK pages=$pages"
}} finally {{
    $saveChanges = 0
    if ($doc -ne $null) {{ $doc.Close([ref]$saveChanges) }}
    $word.Quit([ref]$saveChanges)
}}
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return [f"Microsoft Word COM open/export failed: {proc.stderr.strip() or proc.stdout.strip()}"], 0

    m = re.search(r"WORD_COM_OK pages=(\d+)", proc.stdout)
    if not m:
        return [f"Microsoft Word COM unexpected output: {proc.stdout.strip()}"], 0
    pages = int(m.group(1))
    errs = []
    if pages < 18:
        errs.append(f"Rendered Word page count = {pages} (< 18 pages)")
    if not pdf_path.exists() or pdf_path.stat().st_size < 10000:
        errs.append(f"PDF export missing or empty: {pdf_path}")
    return errs, pages


def main() -> int:
    md_path = Path("outputs/Bao_cao_Giua_Ky_Document_Scanner.md")
    docx_path = Path("outputs/Bao_cao_Giua_Ky_Document_Scanner.docx")
    pdf_path = Path("outputs/Bao_cao_Giua_Ky_Document_Scanner.pdf")

    errors = []
    errors.extend(audit_markdown(md_path))
    docx_errs, media_cnt = audit_docx(docx_path)
    errors.extend(docx_errs)

    pages = 0
    if not docx_errs:
        com_errs, pages = audit_word_com(docx_path, pdf_path)
        errors.extend(com_errs)

    if errors:
        print(f"REPORT AUDIT FAILED ({len(errors)} errors):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(
        f"REPORT AUDIT PASSED: 0 errors (MD + DOCX OOXML + {media_cnt} embedded figures + Word COM clean open, {pages} rendered pages, PDF exported)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

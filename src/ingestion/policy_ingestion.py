"""Improved policy PDF ingestion and section-aware chunking."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "policy" / "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "policy_chunks.json"
SECTION_HEADINGS = ["DEFINITIONS", "EXTENSIONS", "SCOPE OF COVER", "WHAT WE COVER", "WHAT WE EXCLUDE"]

def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("", "•").replace("", "•").replace("�", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_noise(line: str) -> bool:
    line = line.strip()
    if not line or re.fullmatch(r"\d+", line): return True
    patterns = [r"^UNIVERSAL SOMPO GENERAL INSURANCE CO LTD$",
                r"^CSC[-–]\s*Individual Health Insurance[-–]\s*Policy Wording$",
                r"^UNIHLIP"]
    return any(re.search(p, line, re.I) for p in patterns)

def detect_section(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line).strip().upper()
    for heading in SECTION_HEADINGS:
        if normalized == heading: return heading
    return None

def extract_page_units(page_text: str) -> list[str]:
    lines = [clean_text(x) for x in page_text.splitlines()]
    lines = [x for x in lines if not is_noise(x)]
    units, current = [], []
    for line in lines:
        section = detect_section(line)
        if section:
            if current: units.append(clean_text(" ".join(current))); current=[]
            units.append(section); continue
        if line.startswith(("•", "-", "–")) or re.match(r"^(?:\d{1,2}\.|[a-z]\)|[ivx]+\))\s+", line, re.I):
            if current: units.append(clean_text(" ".join(current))); current=[]
        current.append(line)
    if current: units.append(clean_text(" ".join(current)))
    return [u for u in units if len(u) >= 15]

def build_chunks(pdf_path: Path, max_chars: int = 1100) -> list[dict[str, Any]]:
    doc = fitz.open(pdf_path)
    chunks=[]; section="UNSPECIFIED"; counter=0
    for page_no, page in enumerate(doc, 1):
        units=extract_page_units(page.get_text("text"))
        current=[]; length=0
        def flush():
            nonlocal current,length,counter
            if not current: return
            text=clean_text(" ".join(current))
            if not text: current=[]; length=0; return
            counter += 1
            safe=re.sub(r"[^a-z0-9]+","_",section.lower()).strip("_")
            chunks.append({"chunk_id":f"policy_p{page_no:02d}_{safe}_{counter:03d}",
                           "source":pdf_path.name,"page":page_no,"section":section,"text":text})
            current=[]; length=0
        for unit in units:
            detected=detect_section(unit)
            if detected:
                flush(); section=detected; continue
            if current and length+len(unit)+1 > max_chars: flush()
            current.append(unit); length += len(unit)+1
        flush()
    doc.close(); return chunks

def ingest_policy(pdf_path: Path=PDF_PATH, output_path: Path=OUTPUT_PATH):
    if not pdf_path.exists(): raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")
    chunks=build_chunks(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"source":pdf_path.name,"chunk_count":len(chunks),"chunks":chunks},indent=2,ensure_ascii=False),encoding="utf-8")
    print("="*70); print("POLICY INGESTION COMPLETE"); print("="*70)
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {output_path}")
    counts={}
    for c in chunks: counts[c["section"]]=counts.get(c["section"],0)+1
    print("Section distribution:")
    for k,v in sorted(counts.items()): print(f"  {k}: {v}")
    return chunks

if __name__ == "__main__": ingest_policy()

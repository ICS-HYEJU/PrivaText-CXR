"""
Eval_metric/text_utils.py  —  Shared report text preprocessing
==============================================================
Single source of truth for turning a radiology report into the prompt text used
BOTH for generation (conditioning) and for CLIPScore evaluation, so the two are
always consistent.

extract_findings_impression() keeps only the FINDINGS and IMPRESSION sections
(the clinically salient content), mirroring Data/mimic_cxr._parse_report. Falls
back to the full text when neither section is present (e.g. short labels), so a
prompt is never empty.
"""

import re

_SECTION_RE = re.compile(
    r"(FINDINGS|IMPRESSION)\s*:(.*?)(?=\n[A-Z ]+:|$)",
    re.IGNORECASE | re.DOTALL,
)

# The report-section modes selectable via --text_mode / --clip_text_mode.
SECTION_MODES = ["FINDINGS", "IMPRESSION", "FINDINGS/IMPRESSION", "FULL"]


def extract_report_sections(text: str, mode: str = "FINDINGS/IMPRESSION") -> str:
    """
    Reduce a radiology report to the chosen sections.
        FINDINGS            -> only the FINDINGS section
        IMPRESSION          -> only the IMPRESSION section
        FINDINGS/IMPRESSION -> both sections (default)
        FULL                -> the whole text, unchanged
    Falls back to the full text when the requested section(s) are absent, so a
    prompt is never empty.
    """
    text = text or ""
    if mode == "FULL":
        return text.strip()
    if mode == "FINDINGS":
        keys = ("findings",)
    elif mode == "IMPRESSION":
        keys = ("impression",)
    else:
        keys = ("findings", "impression")
    sections = {name.lower(): content.strip()
                for name, content in _SECTION_RE.findall(text)}
    parts = [f"{k.upper()}: {sections[k]}" for k in keys if sections.get(k)]
    return " ".join(parts).strip() or text.strip()


def extract_findings_impression(text: str) -> str:
    """Backward-compatible alias for the FINDINGS/IMPRESSION mode."""
    return extract_report_sections(text, "FINDINGS/IMPRESSION")

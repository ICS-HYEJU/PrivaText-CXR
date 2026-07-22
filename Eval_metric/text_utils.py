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


def extract_findings_impression(text: str) -> str:
    """Return 'FINDINGS: ... IMPRESSION: ...' or the original text if absent."""
    text = text or ""
    sections = {name.lower(): content.strip()
                for name, content in _SECTION_RE.findall(text)}
    if not sections:
        return text.strip()
    parts = [f"{k.upper()}: {sections[k]}"
             for k in ("findings", "impression") if sections.get(k)]
    return " ".join(parts).strip() or text.strip()

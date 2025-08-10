# app_assess_prompt_llm_only.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json, re

# ---- LLM is mandatory ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

app = FastAPI(title="MATNR Assessment & LLM Prompt (Agentic, no fallback)")

# ====== Models ======
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    line: Optional[int] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: int
    end_line: int
    code: str
    matnr_findings: Optional[List[Finding]] = Field(default=None)

# ====== Agent (planner + prompt composer) ======
def plan_tasks(unit: Unit) -> Dict[str, Any]:
    findings = unit.matnr_findings or []
    sev_counts, type_counts, lines = {}, {}, []
    for f in findings:
        sev = (f.severity or "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        it = f.issue_type or "Unknown"
        type_counts[it] = type_counts.get(it, 0) + 1
        if f.line is not None:
            lines.append(f.line)
    return {
        "meta": {
            "program": unit.pgm_name,
            "include": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "range": {"start_line": unit.start_line, "end_line": unit.end_line},
        },
        "stats": {
            "findings_count": len(findings),
            "severity_counts": sev_counts,
            "issue_type_counts": type_counts,
            "lines": sorted(set(lines)),
        }
    }

def build_llm_message(unit: Unit, plan: Dict[str, Any]) -> str:
    findings_json = json.dumps([f.model_dump() for f in (unit.matnr_findings or [])], ensure_ascii=False, indent=2)
    code = unit.code
    MAX = 20000
    if len(code) > MAX:
        code = code[:MAX] + "\n*TRUNCATED*"

    return f"""
You are a senior ABAP reviewer and modernization planner.

Goal:
1) Convert 'matnr_findings' into a concise human-readable **assessment** paragraph for a reporting file (no code changes now).
   Summarize the risks, affected lines, and why they matter re: S/4HANA 40-char MATNR (SAP Note 2215424).
2) Produce a **remediation LLM prompt** to be used later. The prompt must:
   - Reference the unit metadata (program/include/unit/lines).
   - Ask for minimal, behavior-preserving ECC-safe changes (no 7.4+ syntax) focused strictly on MATNR length/compatibility.
   - Require output JSON with keys: original_code, remediated_code, changes[] (line, before, after, reason).
   - No business logic changes, no suppressions, no pseudo-comments.

Return ONLY strict JSON with keys:
{{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt to use later>"
}}

Unit metadata:
- Program: {unit.pgm_name}
- Include: {unit.inc_name}
- Unit type: {unit.type}
- Unit name: {unit.name}
- Start line: {unit.start_line}
- End line: {unit.end_line}

ABAP code (verbatim; may be truncated):
{code}
matnr_findings (JSON):
{findings_json}
""".strip()

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    msg = build_llm_message(unit, plan_tasks(unit))
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a precise ABAP remediation planner that outputs strict JSON only."},
                {"role": "user", "content": msg}
            ],
        )
        content = resp.choices[0].message.content.strip()
        # parse strict JSON; if the model returns extra text, try to extract JSON block
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("Model did not return JSON.")
            return json.loads(m.group(0))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ====== API ======
@app.post("/assess-and-prompt")
def assess_and_prompt(units: List[Unit]) -> List[Dict[str, Any]]:
    """
    Input: array of units (with optional matnr_findings[]).
    Output: same array, replacing 'matnr_findings' with:
      - 'assessment' (string)
      - 'llm_prompt' (string)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"]  = llm_out.get("llm_prompt", "")
        obj.pop("matnr_findings", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}


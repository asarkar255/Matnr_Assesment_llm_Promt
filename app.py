# app_assess_prompt_langchain_min.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json

# ---- LLM is mandatory (no fallback) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="MATNR Assessment & LLM Prompt (LangChain, no fallback, minimal input)")

# ====== Models matching your minimal JSON ======
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None
    # line/start_line/end_line/code are intentionally optional/omitted here

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    # optional fields (not present in your sample)
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""
    matnr_findings: Optional[List[Finding]] = Field(default=None)

# ====== Planning helper (agentic-lite) ======
def summarize_findings(unit: Unit) -> Dict[str, Any]:
    findings = unit.matnr_findings or []
    sev_counts, type_counts = {}, {}
    for f in findings:
        sev = (f.severity or "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        it = f.issue_type or "Unknown"
        type_counts[it] = type_counts.get(it, 0) + 1
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "range": {"start_line": unit.start_line or 0, "end_line": unit.end_line or 0},
        "stats": {
            "count": len(findings),
            "severity_counts": sev_counts,
            "issue_type_counts": type_counts,
        }
    }

# ====== LangChain prompt & chain ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

Goal:
1) Turn 'matnr_findings' into a concise human-readable assessment paragraph for a reporting file (no code changes now).
   Summarize risks and why they matter for S/4HANA 40-char MATNR (SAP Note 2215424).
2) Produce a remediation LLM prompt to be used later. The prompt must:
   - Reference the unit metadata (program/include/unit/lines if provided).
   - Ask for minimal, behavior-preserving ECC-safe changes (no 7.4+ syntax) focused strictly on MATNR length/compatibility.
   - Require output JSON with keys: original_code, remediated_code, changes[] (line, before, after, reason).
   - No business logic changes, no suppressions, no pseudo-comments.

Return ONLY strict JSON with keys:
{{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt to use later>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Start line: {start_line}
- End line: {end_line}

ABAP code (optional; may be empty):
{code}
Planning summary (agentic):
{plan_json}

matnr_findings (JSON):
{findings_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in (unit.matnr_findings or [])], ensure_ascii=False, indent=2)
    plan = summarize_findings(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    try:
        return chain.invoke(
            {
                "pgm_name": unit.pgm_name,
                "inc_name": unit.inc_name,
                "unit_type": unit.type,
                "unit_name": unit.name or "",
                "start_line": unit.start_line or 0,
                "end_line": unit.end_line or 0,
                "code": unit.code or "",
                "plan_json": plan_json,
                "findings_json": findings_json,
            }
        )
    except Exception as e:
        # hard fail (no fallback)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ====== API ======
@app.post("/assess-and-prompt")
def assess_and_prompt(units: List[Unit]) -> List[Dict[str, Any]]:
    """
    Input: array of units (with matnr_findings[]).
    Output: same array, replacing 'matnr_findings' with:
      - 'assessment' (string)
      - 'llm_prompt' (string)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("matnr_findings", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}

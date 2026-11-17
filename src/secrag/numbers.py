"""Deterministic numeric checks used by the Verifier.

LLMs are bad at noticing that "$94.9 billion" and a table cell "94,930" (in
millions) are the same number, and even worse at spotting a wrong digit. So
before trusting the LLM's "SUPPORTED" verdict, every number in a claim must be
found in the cited evidence (or the XBRL facts), allowing for unit scaling and
the rounding implied by how many decimals the claim uses. Derived numbers
(growth rates, margins) must come with an explicit `calculation`, which is
evaluated safely and whose operands must themselves be grounded.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}
_NUM_RE = re.compile(
    r"(?<![\w.])\(?\$?\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?\)?"
    r"(?:\s?(?P<unit>%|percent\b|(?:thousand|million|billion|trillion)s?\b|bn\b|mn\b|[kmb]\b))?",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(
    r"\b10-[KQ]\b|\bItem\s+\d+[A-C]?\b|\b(?:FY|Q)\s?\d{1,4}\b|\bE\d+\b|\b\d{4}-\d{2}-\d{2}\b|\[\d+\]",
    re.IGNORECASE,
)
_TRIVIAL_OPERANDS = {1.0, 100.0, 1000.0, 1e6, 1e9}


@dataclass(frozen=True)
class Num:
    raw: float  # as written
    scale: float  # from unit words ("billion" -> 1e9); 1 if none
    pct: bool
    decimals: int
    text: str

    @property
    def value(self) -> float:
        return self.raw * self.scale


def extract_numbers(text: str, skip_years: bool = True) -> list[Num]:
    text = _NOISE_RE.sub(" ", text)
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        int_part = m.group("int")
        dec = m.group("dec") or ""
        raw = float(f"{int_part.replace(',', '')}.{dec}" if dec else int_part.replace(",", ""))
        unit = (m.group("unit") or "").lower()
        pct = unit in ("%", "percent")
        scale = 1.0
        if unit and not pct:
            scale = _SCALES.get(unit, _SCALES.get(unit.rstrip("s"), 1.0))
        is_year = (
            skip_years
            and not dec
            and "," not in int_part
            and not unit
            and "$" not in m.group(0)
            and 1900 <= raw <= 2100
        )
        if is_year:
            continue
        out.append(Num(raw, scale, pct, len(dec), m.group(0).strip()))
    return out


def _close(a: float, b: float, abs_tol: float, rel_tol: float = 1e-4) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def is_grounded(claim: Num, evidence: list[Num]) -> bool:
    """True if `claim` matches some evidence number up to unit scale and rounding."""
    half_ulp = 0.5 * 10 ** (-claim.decimals)
    if claim.pct:
        return any(_close(claim.raw, e.raw, half_ulp + 1e-9, 0.0) for e in evidence)
    target = claim.value
    for s in (1.0, 1e3, 1e6, 1e9):
        c = target / s
        tol = half_ulp * claim.scale / s
        for e in evidence:
            if e.pct:
                continue
            if _close(c, e.raw, tol) or (s == 1.0 and _close(target, e.value, tol)):
                return True
    return False


def _matches_value(n: Num, x: float) -> bool:
    half_ulp = 0.5 * 10 ** (-n.decimals)
    return _close(x, n.raw, half_ulp + 1e-9, 1e-3) or _close(x, n.value, half_ulp * n.scale, 1e-3)


# ------------------------------------------------------------------ calculations
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)


def _clean_expr(expr: str) -> str:
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    return expr.replace("$", "").replace("%", "").replace("×", "*").replace("−", "-")


def safe_eval(expr: str) -> float:
    """Evaluate + - * / arithmetic only. Raises ValueError on anything else."""
    tree = ast.parse(_clean_expr(expr), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants allowed")
    return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))


def check_calculation(claim_text: str, expr: str, evidence: list[Num]) -> tuple[bool, str, float | None]:
    try:
        result = safe_eval(expr)
    except Exception as e:  # noqa: BLE001
        return False, f"invalid calculation '{expr}': {e}", None
    claim_nums = extract_numbers(claim_text)
    if not any(_matches_value(n, result) for n in claim_nums):
        return False, f"calculation '{expr}' = {result:,.4g}, which does not match any number in the claim", result
    for op in extract_numbers(_clean_expr(expr), skip_years=False):
        if op.raw in _TRIVIAL_OPERANDS:
            continue
        if not is_grounded(op, evidence):
            return False, f"calculation operand {op.text} not found in cited evidence", result
    return True, f"calculation verified ({result:,.4g})", result


def check_claim_numbers(claim_text: str, calculation: str | None, evidence: list[Num]) -> tuple[bool, str]:
    nums = extract_numbers(claim_text)
    calc_result = None
    if calculation:
        ok, msg, calc_result = check_calculation(claim_text, calculation, evidence)
        if not ok:
            return False, msg
    missing = [
        n.text
        for n in nums
        if not is_grounded(n, evidence) and not (calc_result is not None and _matches_value(n, calc_result))
    ]
    if missing:
        return False, "numbers not found in cited evidence: " + ", ".join(missing)
    return True, "all numbers grounded" if nums else "no numbers to check"

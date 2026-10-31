"""Tri-state nutrition-rule decisions and ranking metrics."""
from __future__ import annotations
import ast
import math
from enum import Enum
from typing import Mapping, Sequence

class SafetyStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"

ALIASES = {
    "diabetes": "diabetes", "diabetes_mellitus": "diabetes", "type_2_diabetes": "diabetes", "type2_diabetes": "diabetes",
    "hypertension": "hypertension", "high_blood_pressure": "hypertension",
    "fatty_liver": "steatohepatitis", "fatty_liver_disease": "steatohepatitis", "mash": "steatohepatitis",
    "nash": "steatohepatitis", "steatohepatitis": "steatohepatitis",
    "heart_failure": "heart_failure", "kidney_failure": "kidney_failure",
    "chronic_kidney_disease": "kidney_failure", "chronic_kidney_failure": "kidney_failure", "renal_failure": "kidney_failure", "ckd": "kidney_failure",
}

def parse_profile_conditions(value):
    """[] means explicit no-disease; None means missing or unrecognized profile."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.lower() in {"none", "no_disease", "healthy", "[]"}:
        return []
    output = []
    for raw in text.replace(",", "|").split("|"):
        name = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if not name:
            continue
        if name not in ALIASES:
            return None
        canonical = ALIASES[name]
        if canonical not in output:
            output.append(canonical)
    return output or None

def _value(node, values):
    if isinstance(node, ast.Expression):
        return _value(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        value = values.get(node.id)
        if value is None:
            return None
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return None
        except TypeError:
            pass
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        part = _value(node.operand, values)
        return None if part is None else -part
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, values), _value(node.right, values)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
        except (ArithmeticError, OverflowError):
            return None
    if isinstance(node, ast.BoolOp):
        parts = [_value(part, values) for part in node.values]
        if isinstance(node.op, ast.And):
            if False in parts: return False
            return None if None in parts else all(parts)
        if isinstance(node.op, ast.Or):
            if True in parts: return True
            return None if None in parts else any(parts)
    if isinstance(node, ast.Compare):
        left = _value(node.left, values)
        if left is None:
            return None
        for op, expr in zip(node.ops, node.comparators):
            right = _value(expr, values)
            if right is None:
                return None
            try:
                if isinstance(op, ast.Lt): ok = left < right
                elif isinstance(op, ast.LtE): ok = left <= right
                elif isinstance(op, ast.Gt): ok = left > right
                elif isinstance(op, ast.GtE): ok = left >= right
                elif isinstance(op, ast.Eq): ok = left == right
                elif isinstance(op, ast.NotEq): ok = left != right
                else: raise ValueError("Unsupported comparison operator")
            except (TypeError, ValueError):
                return None
            if not ok:
                return False
            left = right
        return True
    raise ValueError(f"Unsupported medical-rule syntax: {type(node).__name__}")

_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.USub,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BoolOp,
            ast.And, ast.Or, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)

def evaluate_rule(expression: str, nutrition: Mapping[str, object]):
    """Return True/False; return None if a required value is missing."""
    tree = ast.parse(expression, mode="eval")
    if any(not isinstance(node, _ALLOWED) for node in ast.walk(tree)):
        raise ValueError("Unsupported syntax in configurable medical rule")
    value = _value(tree, nutrition)
    return value if isinstance(value, bool) else None

def assess_safety(conditions, nutrition: Mapping[str, object], rules: Mapping[str, object]):
    """All conditions must pass; missing profile or required nutrients give UNKNOWN."""
    if conditions is None:
        return SafetyStatus.UNKNOWN
    if not conditions:
        return SafetyStatus.SAFE
    diseases, unknown = rules.get("diseases", {}), False
    for condition in conditions:
        rule = diseases.get(condition, {})
        if rule.get("alias_of"):
            rule = diseases.get(rule["alias_of"], {})
        expression = rule.get("threshold_condition")
        if not expression:
            unknown = True
            continue
        result = evaluate_rule(expression, nutrition)
        if result is False:
            return SafetyStatus.UNSAFE
        if result is None:
            unknown = True
    return SafetyStatus.UNKNOWN if unknown else SafetyStatus.SAFE

def dcg(relevances: Sequence[float], k: int, safe=None, penalty=5):
    total = 0.0
    for rank, rel in enumerate(relevances[:k], 1):
        gain = 2.0 ** float(rel) - 1.0 if safe is None or safe[rank - 1] else 2.0 ** (-float(penalty)) - 1.0
        total += gain / math.log2(rank + 1)
    return total

def idcg(relevances: Sequence[float], k: int):
    return dcg(sorted((float(x) for x in relevances), reverse=True), k)

def prepare_metric_reference(relevance, status, candidates, ks=(5, 10, 20)):
    """Cache candidate-label denominators shared by all four arms for one user."""
    candidate_ids = set(candidates)
    relevant, eligible = [], []
    for item, value in relevance.items():
        if int(item) in candidate_ids and float(value) > 0:
            relevant.append(float(value))
            if status.get(item, SafetyStatus.UNKNOWN) == SafetyStatus.SAFE:
                eligible.append(float(value))
    return {
        "relevant_count": len(relevant),
        "eligible_relevant_count": len(eligible),
        "ndcg_idcg": {k: idcg(relevant, k) for k in ks},
        "p_ndcg_idcg": {k: idcg(eligible, k) for k in ks},
    }

def safe_ndcg(recommended, relevance, status, candidates, k, penalty=5, reference=None):
    """Per-user metrics; P-NDCG stays unclipped and is undefined for zero eligible IDCG."""
    ranked = list(recommended[:k])
    rel = [float(relevance.get(item, 0.0)) for item in ranked]
    flags = [status.get(item, SafetyStatus.UNKNOWN) == SafetyStatus.SAFE for item in ranked]
    p_dcg = dcg(rel, k, flags, penalty)
    if reference is None:
        relevant, eligible = [], []
        for item in candidates:
            value = float(relevance.get(item, 0.0))
            if value > 0:
                relevant.append(value)
                if status.get(item, SafetyStatus.UNKNOWN) == SafetyStatus.SAFE:
                    eligible.append(value)
        relevant_count, eligible_count = len(relevant), len(eligible)
        standard_idcg, p_idcg = idcg(relevant, k), idcg(eligible, k)
    else:
        relevant_count = reference["relevant_count"]
        eligible_count = reference["eligible_relevant_count"]
        standard_idcg = reference["ndcg_idcg"][k]
        p_idcg = reference["p_ndcg_idcg"][k]
    p_ndcg = p_dcg / p_idcg if p_idcg > 0 else None
    ndcg = dcg(rel, k) / standard_idcg if standard_idcg > 0 else None
    hits = sum(value > 0 for value in rel)
    safe_hits = sum(value > 0 and flag for value, flag in zip(rel, flags))
    count = len(ranked)
    return {
        "Safety@K": sum(flags) / k if count == k else None,
        "SafetyAmongReturned": sum(flags) / count if count else None,
        "FillRate@K": count / k,
        "P-DCG@K": p_dcg, "P-NDCG@K": p_ndcg, "NDCG@K": ndcg,
        "Precision@K": hits / k,
        "Recall@K": hits / relevant_count if relevant_count else None,
        "EligiblePrecision@K": safe_hits / k,
        "EligibleRecall@K": safe_hits / eligible_count if eligible_count else None,
        "Unsafe@1": status.get(ranked[0], SafetyStatus.UNKNOWN) == SafetyStatus.UNSAFE if ranked else None,
        "Unknown@K": sum(status.get(i, SafetyStatus.UNKNOWN) == SafetyStatus.UNKNOWN for i in ranked),
        "RecommendationCount": count, "P_IDCG@K": p_idcg, "P_NDCG_Undefined": p_idcg <= 0,
    }
def experiment_arms():
    """Compare the preference-only baseline with the complete safety-aware system."""
    return [("M1", "m1", False), ("M2", "m2", True)]

def candidate_universe(item_count, excluded):
    excluded = set(map(int, excluded))
    return [item for item in range(item_count) if item not in excluded]
class StatusLookup:
    """Read safety status from one item-aligned vector without allocating a large dict."""
    def __init__(self, values):
        self.values = values

    def get(self, item, default=None):
        try:
            return SafetyStatus(self.values[int(item)])
        except (IndexError, ValueError, TypeError):
            return default

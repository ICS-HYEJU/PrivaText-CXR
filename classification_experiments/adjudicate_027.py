"""Apply the prespecified E027 multi-endpoint decision rule."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "classification_experiments/evidence"

comparison = json.loads((EVIDENCE / "027_counterfactual_conditioning_comparison.json").read_text())
context = json.loads((EVIDENCE / "027_context_shuffle.json").read_text())
prompt = json.loads((EVIDENCE / "027_prompt_response.json").read_text())
baseline_prompt = json.loads((EVIDENCE / "024_prompt_response.json").read_text())["projection"]

candidate = comparison["results"]["counterfactual"]
baseline = comparison["results"]["projection"]
macro = {encoder: {
    "baseline": baseline[encoder]["macro"], "candidate": candidate[encoder]["macro"],
    "delta": candidate[encoder]["macro"] - baseline[encoder]["macro"],
    "ci95": comparison["paired_patient_ci"][encoder]["ci95"],
} for encoder in ("xrv", "medclip")}
context_positive = sum(result["patient_ci95"][0] > 0
                       for result in context["results"].values())
prompt_summary = {}
for encoder in ("xrv", "medclip"):
    base_mean = sum(x["auroc"] for x in baseline_prompt[encoder].values()) / 4
    new_mean = sum(x["auroc"] for x in prompt["counterfactual"][encoder].values()) / 4
    prompt_summary[encoder] = {"baseline_mean": base_mean, "candidate_mean": new_mean,
                               "delta": new_mean - base_mean}

gates = {
    "both_macro_point_estimates_improve": all(macro[e]["delta"] > 0 for e in macro),
    "both_macro_ci_lower_positive": all(macro[e]["ci95"][0] > 0 for e in macro),
    "context_ci_positive_at_least_3_of_4": context_positive >= 3,
    "both_prompt_means_improve": all(prompt_summary[e]["delta"] > 0 for e in prompt_summary),
}
result = {"experiment": "E027", "macro": macro, "context_positive_timesteps": context_positive,
          "prompt": prompt_summary, "gates": gates, "candidate_retained": all(gates.values())}
(EVIDENCE / "027_adjudication.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)

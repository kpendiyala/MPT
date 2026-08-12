from pathlib import Path
import re

path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "weaver/utils/nn/tools.py"
)

text = path.read_text()

old_pattern = re.compile(
    r"eval_metrics=\[\s*"
    r"'roc_auc_score'\s*,\s*"
    r"'roc_auc_score_matrix'\s*,\s*"
    r"'confusion_matrix'\s*"
    r"\]"
)

matches = list(old_pattern.finditer(text))

if not matches:
    if "eval_metrics=[]" in text:
        print("Weaver evaluation metrics already patched.")
        raise SystemExit(0)

    raise RuntimeError(
        "Could not find expected eval_metrics default in Weaver tools.py"
    )

# Patch only the first occurrence: training-time classification evaluation.
# Leave evaluate_onnx alone.
text, count = old_pattern.subn(
    "eval_metrics=[]",
    text,
    count=1,
)

path.write_text(text)

print(f"Patched {path}")
print(f"Replacements: {count}")

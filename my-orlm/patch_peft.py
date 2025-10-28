import re, pathlib
p = pathlib.Path("ORLM/train/finetune.py")
s = p.read_text(encoding="utf-8")
s2 = re.sub(r"\bprepare_model_for_int8_training\b", "prepare_model_for_kbit_training", s)
if s != s2:
    p.write_text(s2, encoding="utf-8")
    print("Patched", p)
else:
    print("No changes needed")
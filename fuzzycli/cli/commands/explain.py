import json
from ...fuzzy.io.fz_parser import parse_fz
from ...fuzzy.model.classifier import Classifier

def _parse_keyvals(kvs):
    data = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        data[k] = float(v)
    return data

def cmd_explain(args):
    kb = parse_fz(args.model)
    clf = Classifier(kb)
    data = _parse_keyvals(args.kv)
    res = clf.explain(data, mode=args.mode if getattr(args, "mode", None) else None,
                      threshold=getattr(args, "threshold", 0.0))
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return
    for oname, rules in res.items():
        print(f"Output: {oname}")
        if rules and isinstance(rules[0], dict) and "_fati_label_strengths" in rules[0]:
            meta = rules[0]["_fati_label_strengths"]
            print("  FATI label strengths:", meta)
            rules = rules[1:]
        for r in rules:
            ants = " AND ".join(f"{a['var']} is {a['label']} (μ={a['mu']:.3f})" for a in r["antecedent"])
            print(f"  R{r['rule_index']}: IF {ants} THEN {r['consequent']['var']} is {r['consequent']['label']}  alpha={r['alpha']:.4f} weight={r['weight']}")

import json
from argparse import Namespace

from .prepare import cmd_prepare
from .learn import cmd_learn
from .show import cmd_show
from .apply import cmd_apply

try:
    from .validate import cmd_validate
except Exception:
    cmd_validate = None

try:
    from .predict import cmd_predict
except Exception:
    cmd_predict = None

try:
    from .explain import cmd_explain
except Exception:
    cmd_explain = None

try:
    import yaml
except ImportError:
    yaml = None

def _ns(d: dict) -> Namespace:
    return Namespace(**d)

def _load_cfg(path: str):
    with open(path, encoding="utf-8") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if not yaml:
                raise RuntimeError("Brak PyYAML. Zainstaluj: pip install pyyaml")
            return yaml.safe_load(f)
        return json.load(f)

def cmd_run(args):
    cfg = _load_cfg(args.config)

    if "prepare" in cfg:
        print("[run] prepare")
        cmd_prepare(_ns(cfg["prepare"]))

    if "learn" in cfg:
        print("[run] learn")
        learn_ns = _ns(cfg["learn"])
        # ← wstrzykujemy całą sekcję 'mf' z config.yaml
        setattr(learn_ns, "mf", cfg.get("mf"))

        # silnik jako fallback (jeśli nie podano w learn)
        eng = (cfg.get("project") or {}).get("engine") or {}
        for k in ("tnorm", "snorm", "mode", "defuzz"):
            if not getattr(learn_ns, k, None):
                setattr(learn_ns, k, eng.get(k))
        cmd_learn(learn_ns)

    if "show" in cfg:
        print("[run] show")
        cmd_show(_ns(cfg["show"]))

    if "apply" in cfg:
        print("[run] apply")
        cmd_apply(_ns(cfg["apply"]))


    if "validate" in cfg:
        if not cmd_validate:
            raise RuntimeError(
                "Sekcja 'validate' jest w configu, ale moduł fuzzycli.cli.validate nie jest dostępny."
            )
        print("[run] validate")
        cmd_validate(_ns(cfg["validate"]))

    if "predict" in cfg:
        if not cmd_predict:
            raise RuntimeError(
                "Sekcja 'predict' jest w configu, ale moduł fuzzycli.cli.predict nie jest dostępny."
            )
        print("[run] predict")
        cmd_predict(_ns(cfg["predict"]))

    if "explain" in cfg:
        if not cmd_explain:
            raise RuntimeError(
                "Sekcja 'explain' jest w configu, ale moduł fuzzycli.cli.explain nie jest dostępny."
            )
        print("[run] explain")
        cmd_explain(_ns(cfg["explain"]))

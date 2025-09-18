from ...fuzzy.model.learner import learn_from_csv, save_kb_to_fz 

def cmd_learn(args):
    # mf z 'run' (args.mf) lub z pliku --mf-config
    mf_cfg = getattr(args, "mf", None)
    mf_cfg_path = getattr(args, "mf_config", "") or ""
    if mf_cfg_path:
        import json
        try:
            import yaml
        except ImportError:
            yaml = None
        with open(mf_cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) if (yaml and mf_cfg_path.endswith((".yml",".yaml"))) else json.load(f)
        mf_cfg = raw.get("mf", raw)

    kb = learn_from_csv(
        csv_path=args.csv,
        terms=getattr(args, "terms", 3),
        partition=getattr(args, "partition", "grid"),
        tnorm=getattr(args, "tnorm", "min") or "min",
        snorm=getattr(args, "snorm", "max") or "max",
        min_weight=getattr(args, "min_weight", 0.0),
        mf_cfg=mf_cfg,  # <<< KLUCZOWE
    )
    save_kb_to_fz(kb, args.out)
    print(f"Saved model to {args.out}")


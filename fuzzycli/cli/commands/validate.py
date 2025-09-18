from ...fuzzy.io.fz_parser import parse_fz

def cmd_validate(args):
    kb = parse_fz(args.model)
    print(f"OK: inputs={len(kb.inputs)}, outputs={len(kb.outputs)}, rules={len(kb.rules)}")
    print(f"tnorm={kb.tnorm}, snorm={kb.snorm}, mode={kb.mode}, defuzz={kb.defuzz}")

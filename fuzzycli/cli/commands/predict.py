from ...fuzzy.io.fz_parser import parse_fz
from ...fuzzy.model.predictor import Predictor

def _parse_keyvals(kvs):
    data = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        data[k] = float(v)
    return data

def cmd_predict(args):
    kb = parse_fz(args.model)
    pred = Predictor(kb)
    data = _parse_keyvals(args.kv)
    out = pred.predict(data)
    for oname, val in out.items():
        print(f"{oname}: {val:.6g}")

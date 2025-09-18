import argparse
from ..argtypes import (
    parse_cols_list, parse_var_col_map,
    MODE_CHOICES, ENC_CHOICES,
    PARTITION_CHOICES, INDUCTION_CHOICES,
    TNORM_DEFAULT, SNORM_DEFAULT,
)
# importy komend:
from .apply import cmd_apply
from .validate import cmd_validate
from .show import cmd_show
from .learn import cmd_learn
from .explain import cmd_explain
from .predict import cmd_predict
from .prepare import cmd_prepare
from .run import cmd_run

def build_parser():
    fmt = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(
        prog="mamdani",
        description=("Mamdani fuzzy CLI – system ekspercki/regułowo-rozmyty "
                     "(prepare → learn → validate/show → predict/explain → apply)"),
        formatter_class=fmt,
        epilog=(
            "Przykłady:\n"
            "  mamdani prepare --csv Iris.csv --in-cols 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm' \\\n"
            "                  --out-col Species --out iris_num.csv --mapping iris_map.json\n"
            "  mamdani learn --csv iris_num.csv --out iris.fz --terms 3 --partition grid --induction wm \\\n"
            "                --mode FIT --tnorm min --snorm max --min-weight 0.0\n"
            "  mamdani validate --model iris.fz\n"
            "  mamdani show --model iris.fz --at sepal_length=5.1 sepal_width=3.5 petal_length=1.4 petal_width=0.2\n"
            "  mamdani predict --model iris.fz sepal_length=5.9 sepal_width=3.0 petal_length=5.1 petal_width=1.8\n"
            "  mamdani explain --model iris.fz sepal_length=5.9 sepal_width=3.0 petal_length=5.1 petal_width=1.8 --json\n"
            "  mamdani apply --model iris.fz --csv Iris.csv --in-cols 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm' \\\n"
            "                --ignore-cols 'Id,Species' --encoding label --out preds.csv\n"
        )
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    # apply
    sp_a = sub.add_parser("apply", help="Zastosuj model do CSV (batch classify)", formatter_class=fmt)
    g_io = sp_a.add_argument_group("Wejście/Wyjście")
    g_io.add_argument("--model", required=True)
    g_io.add_argument("--csv", required=True)
    g_io.add_argument("--out", help="plik wyjściowy CSV (jeśli brak -> stdout)")

    g_sel = sp_a.add_argument_group("Wybór kolumn")
    sp_a.add_argument("--col-map", dest="map", type=parse_var_col_map,
                      help="mapa var=kolumna, np. sepal_length=1 lub sepal_length=SepalLengthCm")
    sp_a.add_argument("--map", dest="map", type=parse_var_col_map, help=argparse.SUPPRESS)  # alias legacy
    sp_a.add_argument("--in-cols", type=parse_cols_list,
                      help="lista kolumn wejściowych (indeksy lub nazwy)")
    sp_a.add_argument("--ignore-cols", type=parse_cols_list,
                      help="lista kolumn do pominięcia (indeksy lub nazwy)")
    sp_a.add_argument("--inputs", type=int, default=None,
                      help="(opcjonalnie) przytnij liczbę wejść modelu do N, gdy używasz --in-cols")

    g_out = sp_a.add_argument_group("Format wyjścia")
    g_out.add_argument("--encoding", choices=ENC_CHOICES, default="label",
                       help="format klas: 'label' | 'decimal' | 'binary'")
    g_out.add_argument("--mode", choices=MODE_CHOICES, help="FIT|FATI; gdy brak, używa trybu z modelu")
    sp_a.set_defaults(func=cmd_apply)
    
    # validate
    sp_v = sub.add_parser("validate", help="Walidacja spójności modelu", formatter_class=fmt)
    sp_v.add_argument("--model", required=True)
    sp_v.set_defaults(func=cmd_validate)

    # show
    sp_s = sub.add_parser("show", help="Pokaż model/MF/reguły; opcj. wartości w punkcie", formatter_class=fmt)
    sp_s.add_argument("--model", required=True)
    sp_s.add_argument("--at", nargs="*")
    sp_s.set_defaults(func=cmd_show)
    sp_s.add_argument("--include-inactive", action="store_true", help="Pokaż również reguły inactive")
    sp_s.add_argument("--fired-only", action="store_true", help="Pokaż tylko reguły, które się odpaliły dla --at")
    sp_s.add_argument("--min-alpha", type=float, default=0.0, help="Próg α dla --fired-only")

    # learn
    sp_l = sub.add_parser("learn", help="Ucz model z (przygotowanego) CSV", formatter_class=fmt)
    sp_l.add_argument("--csv", required=True)
    sp_l.add_argument("--out", required=True)
    sp_l.add_argument("--terms", type=int, default=3, help="liczba MF na zmienną (trójkąty)")
    sp_l.add_argument("--partition", choices=PARTITION_CHOICES, default="grid")
    sp_l.add_argument("--induction", choices=INDUCTION_CHOICES, default="wm")
    sp_l.add_argument("--mode", choices=MODE_CHOICES, default="FIT")
    sp_l.add_argument("--tnorm", default=TNORM_DEFAULT)
    sp_l.add_argument("--snorm", default=SNORM_DEFAULT)
    sp_l.add_argument("--min-weight", type=float, default=0.0, help="próg odrzucenia słabych reguł")
    sp_l.set_defaults(func=cmd_learn)

    # explain
    sp_e = sub.add_parser("explain", help="Wyjaśnij predykcję dla próbki", formatter_class=fmt)
    sp_e.add_argument("--model", required=True)
    sp_e.add_argument("kv", nargs="+")
    sp_e.add_argument("--json", action="store_true")
    sp_e.add_argument("--threshold", type=float, default=0.0)
    sp_e.set_defaults(func=cmd_explain)

    # predict
    sp_p = sub.add_parser("predict", help="Predykcja dla pojedynczej próbki", formatter_class=fmt)
    sp_p.add_argument("--model", required=True)
    sp_p.add_argument("kv", nargs="+", help="key=value pairs")
    sp_p.set_defaults(func=cmd_predict)

    # prepare
    sp_pr = sub.add_parser("prepare", help="Przygotuj numeryczny CSV i mapping ról/kolumn", formatter_class=fmt)
    sp_pr.add_argument("--csv", required=True)
    sp_pr.add_argument("--in-cols", required=True, type=parse_cols_list)
    sp_pr.add_argument("--out-col", required=True)
    sp_pr.add_argument("--num-cols", default="", type=parse_cols_list)
    sp_pr.add_argument("--str-cols", default="", type=parse_cols_list)
    sp_pr.add_argument("--ignore-cols", default="", type=parse_cols_list)
    sp_pr.add_argument("--out", required=True)
    sp_pr.add_argument("--mapping", required=True)
    sp_pr.set_defaults(func=cmd_prepare)

    #run
    sp_run = sub.add_parser("run", help="Uruchom pipeline z pliku konfiguracyjnego")
    sp_run.add_argument("--config", required=True, help="Ścieżka do pliku config.json")
    sp_run.set_defaults(func=cmd_run)

    return ap

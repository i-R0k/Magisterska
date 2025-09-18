from .commands.parser import build_parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    main()

import sys

import simplepyutils as spu


def main():
    result = {}
    for p in sys.argv[1:-1]:
        result.update(fix(spu.load_pickle(p)))
    spu.dump_pickle(result, sys.argv[-1])


def fix(x):
    return {as_str(k): v for k, v in x.items()}


def as_str(x):
    if isinstance(x, str):
        return x
    else:
        return x.decode('utf8')


if __name__ == '__main__':
    main()

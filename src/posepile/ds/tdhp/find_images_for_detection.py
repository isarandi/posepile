import re

import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    for path in spu.sorted_recursive_glob(f'{DATA_ROOT}/3dhp/S*/**/imageSequence/*.jpg'):
        i_frame = int(re.search(r'_(\d+).jpg$', path)[1])
        if i_frame % 5 == 0:
            print(path)


if __name__ == '__main__':
    main()

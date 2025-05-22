import argparse
import tarfile

import imageio.v2 as imageio
import requests
import simplepyutils as spu
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', type=str)
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    spu.initialize(parser)

    url_prefix = 'https://download.is.tue.mpg.de/download.php?domain=rich&sfile='
    response = requests.post(
        f'{url_prefix}{FLAGS.source_file}', stream=True,
        data=dict(username=FLAGS.user, password=FLAGS.password))
    fileobj = response.raw
    tar = tarfile.open(fileobj=fileobj, mode='r|gz')

    with spu.ThrottledPool() as pool:
        for member in spu.progressbar(tar):
            if member.name.endswith(('.bmp', '.png')):
                f = tar.extractfile(member)
                out_path = spu.replace_extension(member.name, '.jpg')
                pool.apply_async(save_jpg, (f.read(), out_path))


def save_jpg(buf, out_path):
    im = imageio.imread(buf)
    spu.ensure_parent_dir_exists(out_path)
    imageio.imwrite(out_path, im, quality=85)


if __name__ == '__main__':
    main()

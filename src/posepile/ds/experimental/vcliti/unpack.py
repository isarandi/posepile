import struct

import lz4.block  # pip install lz4
import numpy as np
import simplepyutils as spu

from posepile.ds.experimental.vcliti.main import VCLITI_ROOT


def main():
    scnz_paths = spu.sorted_recursive_glob(f'{VCLITI_ROOT}/**/*_colordepth.scnz')

    for scnz_path in scnz_paths:
        all_depths = []
        depth_timestamps = []
        rgb_timestamps = []

        with open(scnz_path, 'rb') as file:
            n_frames, = struct.unpack("i", file.read(4))
            version, = struct.unpack("i", file.read(4))
            _ = file.read(1229052)

            for i_frame in range(n_frames):
                i_frame_from_file, = struct.unpack("i", file.read(4))
                assert i_frame == i_frame_from_file

                rgb_timestamp, = struct.unpack("Q", file.read(8))
                depth_timestamp, = struct.unpack("Q", file.read(8))
                n_jpegbytes, = struct.unpack("i", file.read(4))
                jpegbytes = file.read(n_jpegbytes)
                n_depthbytes, = struct.unpack("i", file.read(4))
                depthbytes_compressed = file.read(n_depthbytes)

                depth_timestamps.append(depth_timestamp)
                rgb_timestamps.append(rgb_timestamp)

                impath = scnz_path.replace('_colordepth.scnz', f'/rgb_{i_frame:06d}.jpg')
                spu.ensure_parent_dir_exists(impath)
                with open(impath, 'wb') as jpeg_file:
                    jpeg_file.write(jpegbytes)

                depthbytes_uncompressed = lz4.block.decompress(
                    depthbytes_compressed, uncompressed_size=512 * 424 * 2)
                depth_array = np.frombuffer(
                    depthbytes_uncompressed, dtype=np.uint16).reshape([424, 512])
                all_depths.append(depth_array)

        timestamp_path = scnz_path.replace('_colordepth.scnz', '_timestamps.npz')
        depth_timestamps = np.array(depth_timestamps, dtype=np.uint64)
        rgb_timestamps = np.array(rgb_timestamps, dtype=np.uint64)
        np.savez_compressed(
            timestamp_path, depth_timestamps=depth_timestamps, rgb_timestamps=rgb_timestamps)

        depth_path = scnz_path.replace('_colordepth.scnz', '_depth.npz')
        np.savez_compressed(depth_path, depth=np.stack(all_depths, axis=0))


if __name__ == '__main__':
    main()

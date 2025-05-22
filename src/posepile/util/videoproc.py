import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu


def video_extents(filepath):
    """Returns the video (width, height) as a numpy array, without loading the pixel data."""

    with imageio.get_reader(filepath, 'ffmpeg') as reader:
        return np.asarray(reader.get_meta_data()['source_size'])


def get_fps(filepath):
    with imageio.get_reader(filepath) as reader:
        return reader.get_meta_data()['fps']


def transform(inp_path, out_path, process_frame_fn, **kwargs):
    spu.ensure_parent_dir_exists(out_path)
    with imageio.get_reader(inp_path) as reader:
        fps = reader.get_meta_data()['fps']
        n_frames = num_frames(inp_path)
        with imageio.get_writer(out_path, fps=fps, codec='h264', **kwargs) as writer:
            for frame in spu.progressbar(reader, total=n_frames):
                writer.append_data(process_frame_fn(frame))


def num_frames(path):
    with imageio.get_reader(path, 'ffmpeg') as vid:
        metadata = vid.get_meta_data()
        n = metadata['nframes']
        # check if its integer and not string etc
        if isinstance(n, int):
            return n
        else:
            return int(metadata['duration'] * metadata['fps'])




def video_audio_mux(vidpath_audiosource, vidpath_imagesource, out_video_path):
    import ffmpeg
    video = ffmpeg.input(vidpath_imagesource).video
    audio = ffmpeg.input(vidpath_audiosource).audio
    ffmpeg.output(audio, video, out_video_path, vcodec='copy', acodec='copy').run()

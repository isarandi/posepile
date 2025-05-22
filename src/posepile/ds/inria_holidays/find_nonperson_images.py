import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    detections_all = spu.load_pickle(f'{DATA_ROOT}/inria_holidays/yolov3_person_detections.pkl')
    filenames_without_detection = sorted([
        filename for filename, detections in detections_all.items() if not detections])
    spu.write_file(
        content='\n'.join(filenames_without_detection),
        path=f'{DATA_ROOT}/inria_holidays/non_person_images.txt')


if __name__ == '__main__':
    main()

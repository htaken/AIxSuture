import os.path
import parse
import cv2
import argparse

from datetime import datetime

class Aachen:
    """ Define properties of Aachen dataset"""
    
    #fps = 29.9733
    file_extension = ".mp4"

def extract_frames(video_dir, out_dir, extract_fps, height=270, width=480, ext='.jpg'):  # from Aachen
    '''
    Extracts frames from videos in the folder video_dir, extracts them at a calculated framerate based on the step size,
    rescales them, and places the extracted frames in the out_dir

    :param video_dir: path to folder with videos, relative to project
    :param out_dir: path where to store extracted frames; the date, fps, and video name are appended to the file path
    :param step: step size with which to extract frames, default: 25
    :param height: height to rescale, default: 270
    :param width: width to rescale, default: 480
    :param ext: file extension, default: .jpg
    '''
    #fps = Aachen.fps / step
    for video_file in sorted([v for v in os.listdir(video_dir) if v.endswith(Aachen.file_extension)]):
        print("Starting parse")
        video_id = parse.parse("{}" + Aachen.file_extension, video_file)[0]
        target_dir = os.path.join(out_dir, "frames_{}fps".format(int(extract_fps)), video_id)
        if os.path.exists(target_dir):
            files = [f for f in os.listdir(target_dir) if not f.startswith(".")]
            if len(files) > 0:
                print("Skipping non-empty directory '{}'...".format(target_dir))
                continue

        os.makedirs(target_dir)
        print("Extracting frames at {:.2f} fps from video {} to '{}'...".format(extract_fps, video_id, target_dir))

        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        step = round(video_fps/extract_fps)
        if step < 1:
            print("WARNING: Extracting frame rate is greater than video frame rate. Skipping video")
            continue
        #assert (round(video_fps) == round(Aachen.fps)) # Warning: may cause issues depending on framerates of other videos

        extracted_count = 0
        frame_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and ((frame_no % step) == 0):
                frame = rescale(frame, height, width)
                cv2.imwrite(os.path.join(target_dir, "img_{:05d}".format(extracted_count) + ext), frame)
                extracted_count += 1
            if ret:
                frame_no += 1
            else:
                break

        assert (frame_no == cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()


def rescale(img, height, width=None):
    w = img.shape[1]
    h = img.shape[0]
    assert(h <= w)

    if h != height or (width is not None and w != width):
        if width is None:
            width = height * (w / h)  # keep aspect ratio
        img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evaluation report.")
    parser.add_argument("--data_root", type=str, default="??", help="Where to look for the Aachen phase annotations.")
    parser.add_argument("--out_dir", type=str, default="data", help="Where to store the evaluation report. Optional. Files will be found under <out_dir>/preprocessing/<run>/frames_<frame_rate>fps/<video_name>")
    parser.add_argument("--run", type=str, default=datetime.today().strftime('%Y%m%d'), help="The experimental run to evaluate. Optional.")
    parser.add_argument("--frame_rate",default=1, type=int, help="Framerate at which to extract frames (default 1)")

    args = parser.parse_args()

    if args.data_root == '??':
        print("Please specify the location of the video data directory using the --data_root option.")
    else:
        out_dir = os.path.join(args.out_dir, "preprocessing", args.run)
        extract_frames(args.data_root, out_dir, args.frame_rate)

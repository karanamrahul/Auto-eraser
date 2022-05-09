###############################################################################
#           Welcome to the Auto-eraser demo!                                #
###############################################################################




from helper import *
from inpaint_model import *
from segmentation import *
from mask_inpainting import *
import argparse
import warnings



parser = argparse.ArgumentParser()
parser.add_argument('--video', default='', type=str,
                    help='The location of the Video to be operated on.')


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    videoPath = args.video
    minConfidence = 0.5
    inflation = 5

    if not os.path.exists(videoPath):
        sys.exit("Could not locate video '" + videoPath + "'")

    try:
        os.mkdir('output/')
    except Exception:
        pass

    print("Looking for Magic Words in audio...")
    # objectsToMask = get_objs_to_mask(videoPath)
    objectsToMask = "cat"
    print("Magic Word(s) found. Masking objects from the video...")
    fps = get_masks_video(videoPath, objectsToMask, minConfidence, inflation)

    # print("Masking completed. Painting out masked objects...")
    # inpaint_video(videoPath)

    # print("Inpainting completed. Compiling to video...")
    # compiledPath = frames_to_video(videoPath, fps)
    

    # #remove soundless video
    # os.remove(compiledPath)

    # print("Compiled! to " + compiledPath[:-3] + 'mp4')

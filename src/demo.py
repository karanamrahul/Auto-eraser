###############################################################################
#           Welcome to the Auto-eraser demo!                                #
###############################################################################

"""
 *  MIT License
 *
 *  Copyright (c) 2022 Rahul Karanam
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""

"""
 @file       demo.py
 @author     Rahul Karanam
 @copyright  MIT License
 @brief      This file contains the main function of the Auto-eraser demo.
 
"""

from helper import *
from inpaint_model import *
from segmentation import *
from mask_inpainting import *
import argparse
import warnings
import time



parser = argparse.ArgumentParser()
parser.add_argument('--video', default='', type=str,
                    help='The path to the video file.')



if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    videoPath = args.video
    minConfidence = 0.5 # minimum confidence for a detection to be considered a valid detection
    inflation = 5 # the amount of pixels to inflate the bounding box by when drawing the bounding box

    if not os.path.exists(videoPath):
        sys.exit("The Video Path is not found :-( '" + videoPath + "'")

    try:
        os.mkdir('output/')
    except Exception:
        pass
    start_time = time.time() # start the timer for the entire demo    
    
    counter = count_frames(videoPath)
    objectsToMask = ["bus," for i in range(counter)] # list of objects to mask
    print("Total number of frames in the video : ", counter)
    print("Masking in progress...")
    fps = get_masks_video(videoPath, objectsToMask, minConfidence, inflation) # get the masks for the video
    print("Frames per second : ", fps)

    print("Masking completed --> Now Removing the objects from the video...")
    inpaint_video(videoPath)   # inpaint the video
    print("Inpainting completed --> Now saving the video...")
    print("Removing the objects from the video completed...")
    compiledPath = frames_to_video(videoPath, fps) # compile the frames into a video
    add_audio_to_video(videoPath, compiledPath, fps) # add the audio to the video

    #remove soundless video
    os.remove(compiledPath)
    
    # Uncomment the following line to see the output video in real time #
    #show_output_video(videoPath)


    print("Output is saved to ...... :-) " + compiledPath[:-3] + 'mp4')
    
    print("Total time taken for the demo : ", time.time() - start_time)

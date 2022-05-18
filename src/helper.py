

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
 @file       helper.py
 @author     Rahul Karanam
 @copyright  MIT License
 @brief      This file contains helper functions for the Auto-eraser.
 
"""
# Importing the libraries

import traceback
import numpy as np
import cv2
import time
import os
import copy
import mxnet as mx
from gluoncv import data, utils, model_zoo
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as category_names
import sys
import glob
import natsort
import moviepy.editor  as mpy



def filter_detections(objects_to_mask,class_ids, scores, boxes, masks):
    
    
    """
    @brief: Filters the detections based on the objects to mask.
            It removes the detections that are not in the objects to mask.
            
            Args:
                objects_to_mask: list of objects to mask
                class_ids: list of class ids
                scores: list of scores
                boxes: list of bounding boxes
                masks: list of masks
            Returns:
                class_ids: list of class ids updated
                scores: list of scores updated
                boxes: list of bounding boxes updated
                masks: list of masks updated
    """
    objects_to_be_removed = [] # list of objects to be removed
    for i in objects_to_mask:   # i is the object to be masked
        if i:    # if the object is in the list
            objects_to_be_removed.append(category_names.get(i)) # get the name of the object
            
    obj_idx = []   # indices of objects to be removed

    for i in range(class_ids.size): # loop over all detections
        if objects_to_be_removed.count(int(class_ids[i])) == 0: # if the object is not in the list of objects to be removed
            obj_idx.append(i) # append the index of the object
            
    class_ids = np.delete(class_ids, obj_idx) # delete the objects to be removed from the class_ids
    scores = np.delete(scores, obj_idx) # delete the objects to be removed from the scores
    boxes = np.delete(boxes, obj_idx, 0) # delete the objects to be removed from the boxes
    masks = np.delete(masks, obj_idx, 0) # delete the objects to be removed from the masks
    
    return class_ids, scores, boxes, masks


def transform_detections(masks,original_image,inflate_size):
    
    """
    @brief: Transforms the detections to the desired format for inpainting.
    
    Args:
        masks: list of masks
        original_image: original image
        inflate_size: size of the inflate
        
    Returns:
        img_copy: image before output
        img_output: image after output
    """
     
    img_copy = original_image.copy() # copy the original image
    
    for mask in masks: # loop over all masks 
        k = 0 # k is the row index
        for rows in img_copy: # loop over all rows
            i = 0    # i is the column index
            for element in rows: # loop over all elements
                if (mask[k,i] != 0): # if the element is not black
                    rows[i] = [255, 255, 255] # set the element to white
                i = i + 1 # increment the column index
            k = k + 1 # increment the row index
    img_output = img_copy.copy() # copy the image before output
    k = 0 # k is the row index
    for row in img_copy: # loop over all rows
        i = 0   # i is the column index
        for element in row: # loop over all elements
            if np.array_equal(element, [255, 255, 255]):    # if the element is white (not black)
                if not only_white_px(img_output,i,k): # if the pixel is not only white
                    img_output = cv2.circle(img_output, (i, k), inflate_size, (255,255,255),0) # inflate the pixel
            i = i + 1 # increment the column index
        k = k + 1 # increment the row index
        
    mask_output = img_output.copy() # copy the image output
    
    j = 0 # j is the row index
    for row in mask_output:
        i = 0 # i is the column index
        for element in row:
            if not np.array_equal(element, [255, 255, 255]):
                row[i] = [0, 0, 0]
                
            i = i + 1
        j = j + 1
        
    return img_output, mask_output
    
    
    
    
    

            
        
def only_white_px(img,k,i):
    
    """
    @brief:Checks if the pixel is white or not.

    Args: img: image
        k: row index
        i: column index
        
    Returns: True if the pixel is white, False otherwise

    """
    try:
        if np.array_equal(img[k+1][i], [255, 255, 255]): # if the pixel is white
            return False
    except:
        pass
    try:
        if np.array_equal(img[k-1][i], [255, 255, 255]): # if the pixel is white 
            return False
    except:
        pass

    try:
        if np.array_equal(img[k][i+1], [255, 255, 255]):
            return False
    except:
        pass
    try:    
        if np.array_equal(img[k][i-1], [255, 255, 255]):
            return False
    except:
        pass
    
    return True



class HiddenPrints: # class to hide prints from the console (for debugging) in the Auto-eraser
    def __enter__(self): # enter the class and hide the prints
        self._original_stdout = sys.stdout # save the original stdout in the class
        sys.stdout = open(os.devnull, 'w') # set the stdout to null 

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close() # close the stdout file
        sys.stdout = self._original_stdout # set the stdout to the original stdout (the one before the class was entered)
        

def frames_to_video(video_path,fps):
    
    """
    @brief: Converts the frames to a video.
    
    Args:
        video_path: path to the video
        fps: frames per second
        
    Returns:
        VideoOutput Path: path to the video output
    """
    
    video_name = video_path.split(".")[0].split('/')[-1] # get the video name
    out_dir = "output/" + video_name + "/*" # set the output directory
    names_list = [] # list of frames names
    
    
    for name in [os.path.normpath(x) for x in glob.glob(out_dir)]: # loop over all frames
        if name.split(".")[0][-3:] == "out":
            names_list.append(name) # append the frame name to the list
            
    names_list = natsort.natsorted(names_list) # sort the list
    
    frame_list = [] # list of frames
    size_img = (256,256) # shape of the frames
    
    for file in names_list:
        img = cv2.imread(file) # read the frame
        h,w,c = img.shape # get the shape of the frame
        size_img = (w,h) # set the shape of the frame
        frame_list.append(img) # append the frame to the list
    video_out_name = "output/" + video_name + "_out.avi" # set the output video name
     # create the video writer
    out = cv2.VideoWriter(video_out_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size_img)
    for i in range(len(frame_list)):
        # writing to a image array
        try:
            out.write(frame_list[i])
        except Exception:
            print(traceback.format_exception())
    out.release() # release the video writer
    
    return video_out_name # return the path to the video output
  
   
def count_frames(videoPath):
    """
    @brief: Counts the frames of a video.
    
    Args:
        videoPath: path to the video
        
    Returns:
        Number of frames: number of frames
    
    """
    cap = cv2.VideoCapture(videoPath)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    i = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == False:
            break

        i = i + 1

    # When everything done, release the video capture object
    cap.release()

    return i
def add_audio_to_video(originalVideoPath, outputVideoPath, fps):
    """
    @brief: Adds audio to a video.
    
    Args:
        originalVideoPath: path to the original video
        outputVideoPath: path to the output video
        fps: frames per second
    
    Returns:
        VideoOutput Path: path to the video output with audio
    """
    orig_audio = mpy.AudioFileClip(originalVideoPath)
    with HiddenPrints():
        out_clip = mpy.VideoFileClip(outputVideoPath)
        final_clip = out_clip.set_audio(orig_audio)
        final_clip.write_videofile(outputVideoPath[:-3] + 'mp4', fps=fps)
        out_clip.close()
    return
        
def show_output_video(video_path):
    
    """
    @brief: Shows the output video in multiple windows.
    
    window - 1 - Mask
    window - 2 - Output + Mask
    window - 3 - Input Image + Mask
    
    
    Args:
        video_path: path to the video
    """
    
    
    video_name = video_path.split(".")[0].split('/')[-1] # get the video name
    out_dir = "output/" + video_name + "/*" # set the output directory
    input_list = [] # list of frames names
    mask_list =    [] # list of masks names
    output_list = [] # list of output frames names
    
    
    for name in [os.path.normpath(x) for x in glob.glob(out_dir)]: # loop over all frames
        if name.split(".")[0][-3:] == "out":
            output_list.append(name) # append the frame name to the list
        elif name.split(".")[0][-3:] == "put":
            input_list.append(name)
        elif name.split(".")[0][-3:] == "ask":
            mask_list.append(name)
            
    input_list = natsort.natsorted(input_list) # sort the list
    mask_list   = natsort.natsorted(mask_list) # sort the list
    output_list = natsort.natsorted(output_list) # sort the list
    
    print(len(input_list))
    print(len(output_list))
    print(len(mask_list))
    
    for i in range(len(output_list)):
        
        input = cv2.imread(input_list[i]) # read the input frame
        mask = cv2.imread(mask_list[i]) # read the mask
        output = cv2.imread(output_list[i]) # read the output frame
        a = np.hstack((mask,input, output)) # stack the frames
        cv2.putText(a, "Input + Mask", (490,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,22,125), 2)
        cv2.putText(a, " Mask ", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,122,255), 2)
        cv2.putText(a, "Output", (1000,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,11,122), 2)

        cv2.imshow("output", a) # show the frames
        cv2.waitKey(50) # wait 1ms
    cv2.destroyAllWindows() # destroy all windows
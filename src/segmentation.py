########################################################################################################################
#                            SEGMENTATION FUNCTIONS                                                                    # 
########################################################################################################################        

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
 @file       segmentation.py
 @author     Rahul Karanam
 @copyright  MIT License
 @brief      This file contains the main function for the segmentation of the video.
 
"""

from helper import *
from coco_names import *
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)



def get_masks_video(path, objects_to_mask, min_confidence, inflation):
    
    """
    @brief: Generate masks from a video
    Args:
        path: path to video
        objects_to_mask: list of objects to mask
        min_confidence: minimum confidence to mask
        inflation: inflation factor
    Returns:
        fps: frames per second

    """
    
    with HiddenPrints():
        model_mask_rcnn = model_zoo.get_model('mask_rcnn_fpn_resnet101_v1d_coco', pretrained=True) # get model
    video = cv2.VideoCapture(path) # open video
    frames_per_second = video.get(cv2.CAP_PROP_FPS) # get number of frames
    
    if (video.isOpened() == False):
        print("Error opening video stream or file")
        
    
    img_name = path.split(".")[0].split('/')[-1] # get name of the video
    
    try:
        os.mkdir('output/' + img_name) # create output directory
    except Exception:
        pass
    k = 0 
    while (video.isOpened()):
        ret,img  = video.read() # read frame
        
        if ret == False:
            break
        get_masks_image(img, objects_to_mask[k], img_name, k, model_mask_rcnn, min_confidence, inflation) # generate masks
        
        k = k + 1 # increment frame
        print("Frame: " + str(k))
    video.release() # release video
    
    return frames_per_second
    

def get_masks_image(image, objects_to_mask, fileName,frame_id,model_mask,thres, inflation):
    """
    @brief: Generate masks from an image
    Args:
        image: image to be operated on  (numpy array)
        objects_to_mask: list of objects to mask
        fileName: name of the file
        frame_id: frame id
        model_mask: model to use
        thres: threshold to use
        inflation: inflation factor
    Returns:
        input image  (numpy array)
        mask image (numpy array
    """
    
    
    h, w, c = image.shape # height, width, channels
    
    image = mx.nd.array(image)
    a,org_img = data.transforms.presets.rcnn.transform_test(image, short=256)   # resize image to 256x256
    
    if objects_to_mask == "": # if no objects to mask
        print("No objects to mask")
        img_name = fileName + str(frame_id)
        cv2.imwrite('output/' + fileName + '/' + img_name + '_input.png', org_img) # save input image
        img_black = copy.deepcopy(org_img) # create black image 
        j = 0 # row
        for row in img_black: # row in image to be masked (black)
            i =  0 # column
            for element in row: # for each pixel in the row
                row[i] = [0,0,0] # set all pixels to black (0,0,0)
                i = i + 1 # increment column
            j = j + 1 # increment row
        cv2.imwrite('output/' + fileName + '/' + img_name + '_mask.png', img_black)
        
        
    else: # if objects to mask
        print("Masking objects " + str(objects_to_mask))
        objects_to_mask = objects_to_mask.split(",") # split objects to mask
        for i in range(len(objects_to_mask)): # for each object to mask
            objects_to_mask[i] = objects_to_mask[i].strip() # remove whitespace
            
            
        class_ids,scores, boxes , masks = [b[0].asnumpy() for b in model_mask(a)] # inference is done here
        
        for i in range(scores.size): # for each object in the image (scores.size = number of objects)
            if (scores[i] > thres) & (scores[i] < 0.5): # if confidence is above threshold and below 0.5
                scores[i] = 0.5 # set confidence to 0.5 (to avoid masking too much)
                
        class_ids,scores,boxes,masks = filter_detections(objects_to_mask,class_ids,scores,boxes,masks) # filter detections based on objects to mask
        
        w,h = org_img.shape[1],org_img.shape[0] # get image dimensions
        
        masks = utils.viz.expand_mask(masks,boxes,(w,h),scores) # expand masks to the size of the image (w,h)
        
        img_input , masked_img = transform_detections(masks,org_img,inflation) # transform detections to the size of the image (w,h)
        
        cv2.imwrite('output/' + fileName + '/' + fileName + str(frame_id) + '_input.png', img_input) # save input image
        cv2.imwrite('output/' + fileName + '/' + fileName + str(frame_id) + '_mask.png', masked_img) # save mask image

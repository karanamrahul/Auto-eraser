from helper import *
from coco_names import *



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
    pass
    
    
    

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
    
    image = np.array(image)
    a,org_img = data.transforms.presets.rcnn.transform_test(image, short=256)   # resize image to 256x256
    
    if objects_to_mask == "": # if no objects to mask
        img_name = fileName + str(frame_id)
        cv2.write('output/' + fileName + '/' + img_name + '_input.png', org_img) # save input image
        img_black = np.zeros((h,w,c), np.uint8) # create black image 
        j = 0 # row
        for row in img_black: # row in image to be masked (black)
            i =  0 # column
            for element in row: # for each pixel in the row
                row[i] = [0,0,0] # set all pixels to black (0,0,0)
                i = i + 1 # increment column
            j = j + 1 # increment row
        cv2.imwrite('output/' + fileName + '/' + img_name + '_mask.png', img_black)
        
        
    else: # if objects to mask
        objects_to_mask = objects_to_mask.split(",") # split objects to mask
        for i in range(len(objects_to_mask)): # for each object to mask
            objects_to_mask[i] = objects_to_mask[i].strip() # remove whitespace
            
            
        class_ids,scores, boxes , masks = [b[0].asnumpy() for b in model_mask(a)] # inference is done here
        
        for i in range(scores.size): # for each object in the image (scores.size = number of objects)
            if (scores[i] > thres) and (scores[i] < 0.5): # if confidence is above threshold and below 0.5
                scores[i] = 0.5 # set confidence to 0.5 (to avoid masking too much)
                
        class_ids,scores,boxes,masks = filter_detections(objects_to_mask,class_ids,scores,boxes,masks) # filter detections based on objects to mask
        
        w,h = org_img.shape[:2] # get image dimensions
        
        masks = utils.viz.expand_mask(masks,boxes,(w,h),scores) # expand masks to the size of the image (w,h)
        
        img_input , masked_img = tranform_detections(masks,org_img,inflation) # transform detections to the size of the image (w,h)
        
        cv2.write('output/' + fileName + '/' + fileName + str(frame_id) + '_input.png', img_input) # save input image
        cv2.write('output/' + fileName + '/' + fileName + str(frame_id) + '_mask.png', masked_img) # save mask image
        
          
    
########################################################################################################################
#                            SEGMENTATION FUNCTIONS                                                                    # 
########################################################################################################################        



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
                
        class_ids,scores,boxes,masks = remove_undesired_objects(objects_to_mask,class_ids,scores,boxes,masks) # filter detections based on objects to mask
        
        w,h = org_img.shape[1],org_img.shape[0] # get image dimensions
        
        masks = utils.viz.expand_mask(masks,boxes,(w,h),scores) # expand masks to the size of the image (w,h)
        
        img_input , masked_img = create_input_for_inpainting(masks,org_img,inflation) # transform detections to the size of the image (w,h)
        
        cv2.imwrite('output/' + fileName + '/' + fileName + str(frame_id) + '_input.png', img_input) # save input image
        cv2.imwrite('output/' + fileName + '/' + fileName + str(frame_id) + '_mask.png', masked_img) # save mask image
def generate_masks_from_video(videoPath, objectsToMask, minConfidence, inflation):

    with HiddenPrints():
        mrcnn = model_zoo.get_model("mask_rcnn_fpn_resnet101_v1d_coco", pretrained=True)

    cap = cv2.VideoCapture(videoPath)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed

    frameName = videoPath.split(".")[0].split('/')[-1]

    try:
        os.mkdir('output/' + frameName)
    except Exception:
        pass

    i = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == False:
            break

        generate_mask_from_image(frame, objectsToMask[i], frameName, i, mrcnn, minConfidence, inflation)

        i = i + 1

    # When everything done, release the video capture object
    cap.release()

    return fps


def generate_mask_from_image(frame, objectsToMask, frameName, frameIndex, model, THRESHOLD, inflation):
    ## Inputs:
    ##       image_path: string path to image to be operated on
    ##       objects_to_remove: list of object names to be masked
    ##       model: the MaskRCNN model
    ## Outputs:
    ##       input image
    ##       mask image
    ## both images serve as input to the inpainting model

    height, width, channels = frame.shape

    frame = mx.nd.array(frame)
    x, orig_img = data.transforms.presets.rcnn.transform_test(frame, short=256)

    #print(objectsToMask)
    if objectsToMask == "":
        frameNameFrameIndex = frameName + str(frameIndex)
        cv2.imwrite('output/' + frameName + '/' + frameNameFrameIndex + '_input.png', orig_img)
        blackFrame = copy.deepcopy(orig_img)
        j = 0
        for row in blackFrame:
            i = 0
            for element in row:
                row[i] = [0,0,0]
                i = i + 1
            # print(row)
            j = j + 1
        cv2.imwrite('output/' + frameName + '/' + frameNameFrameIndex + '_mask.png', blackFrame)

    else:
        objectsToMask = objectsToMask.split(",")
        for i in range(len(objectsToMask)):
            objectsToMask[i] = objectsToMask[i].strip()

        #print(objectsToMask)
        #print(index_dict.get(objectsToMask[0]))
        # inference is done here
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in model(x)]

        for i in range(scores.size):
            if (scores[i] > THRESHOLD) & (scores[i] < 0.5):
                scores[i] = 0.51

        ids, scores, bboxes, masks = remove_undesired_objects(objectsToMask, ids, scores, bboxes, masks)

        width, height = orig_img.shape[1], orig_img.shape[0]
        masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)

        # masked_img = create_mask_img(masks, orig_img)
        input_img, masked_img = create_input_for_inpainting(masks, orig_img, inflation)

        frameNameFrameIndex = frameName + str(frameIndex)

        # input_img, masked_img = cv2.resize(input_img, desiredOutputSize ), cv2.resize(masked_img, desiredOutputSize)

        cv2.imwrite('output/' + frameName + '/' + frameNameFrameIndex + '_mask.png', masked_img)
        cv2.imwrite('output/' + frameName + '/' + frameNameFrameIndex + '_input.png', input_img)
     
          
def remove_undesired_objects(objects_to_remove, ids, scores, bboxes, masks):
    desired_object_ids = []
    for obj in objects_to_remove:
        if obj:
            desired_object_ids.append(category_names.get(obj))

    #print(desired_object_ids)

    ids_to_remove = []

    for i in range(ids.size):
        if desired_object_ids.count(int(ids[i])) == 0:
            ids_to_remove.append(i)

    #print(ids_to_remove)

    ids = np.delete(ids, ids_to_remove)
    scores = np.delete(scores, ids_to_remove)
    bboxes = np.delete(bboxes, ids_to_remove, 0)
    masks = np.delete(masks, ids_to_remove, 0)
    return ids, scores, bboxes, masks


def create_input_for_inpainting(masks, original_image, INFLATE_SIZE):
    img_pre_output = original_image.copy()
    for mask in masks:
        j = 0
        for row in img_pre_output:
            i = 0
            for element in row:
                if (mask[j, i] != 0):
                    row[i] = [255, 255, 255]
                i = i + 1
            # print(row)
            j = j + 1
    img_output = img_pre_output.copy()
    j = 0
    for row in img_pre_output:
        i = 0
        for element in row:
            if np.array_equal(element, [255, 255, 255]):
                if not only_one_white_pixel(img_pre_output, i, j):
                    img_output = cv2.circle(img_output, (i, j), INFLATE_SIZE, (255, 255, 255), 0)
            i = i + 1
        j = j + 1

    mask_output = img_output.copy()
    j = 0
    for row in mask_output:
        i = 0
        for element in row:
            if not np.array_equal(element, [255, 255, 255]):
                row[i] = [0, 0, 0]
            i = i + 1
        # print(row)
        j = j + 1

    return img_output, mask_output


def only_one_white_pixel(img, i, j):
    try:
        if np.array_equal( img[j+1][i],  [255,255,255]):
            return False
    except:
        pass
    try:
        if np.array_equal( img[j-1][i],  [255,255,255]):
            return False
    except:
        pass
    try:
        if np.array_equal( img[j][i+1],  [255,255,255]):
            return False
    except:
        pass
    try:
        if np.array_equal( img[j][i-1],  [255,255,255]):
            return False
    except:
        pass

    return True
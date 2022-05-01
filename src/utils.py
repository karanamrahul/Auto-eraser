import cv2
from cv2 import merge
import numpy as np
import random
import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import speech_recognition as sr

"""

We have 91 object classes and we need to create different color mask for each class
instance.

"""
COLORS = np.random.uniform(0,255,size = (len(coco_names),3))



def get_outputs(image , model , threshold):
    
    """
    @brief: This function returns the output from the R-Mask CNN
            i.e masks,boxes and labels (Predicted)
            
    Args : 
    
         image : This is the image which will be used for segmentation
         model : Mask R-CNN model
         threshold : Threshold for the model ( pre-defined score below which we will ignore the mask)
    
    Returns : 
    
        masks : This is the mask for the image
        boxes : This is the predicted bounding box for the image
        labels : This is the predicted label for the image
        
    """
    with torch.no_grad():
         # Forward pass through the model
        outputs = model(image)
        
    # Now we need to get the masks,labels and the boxes from the outputs
    
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    
    # We apply the threshold to the scores
    threshold_preds_idx = [scores.index(i) for i in scores if i > threshold]
    threshold_preds_count = len(threshold_preds_idx)
    
    # Now we get the masks which have the score above the threshold(0.5) as to get a soft mask
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    
    # We select only the masks which have the score above the threshold
    masks = masks[:threshold_preds_count]
    
    # Get the boxes which have the score above the threshold
    boxes = [[(int(i[0]),int(i[1])),(int(i[2]),int(i[3]))] for i in outputs[0]['boxes']][:threshold_preds_count]
    
    
    # Get the labels which have the score above the threshold
    labels = [coco_names[i] for i in outputs[0]['labels']]
    # Add group-id to the same class as the labels
    
      # Assign different group id to the same class label
    labels = [i + '-' + str(j) for j,i in enumerate(labels)]
    return masks , boxes , labels


def draw_segmentation(image,masks,boxes,labels):
    """
    
    @brief: This function will draw the segmentation on the image
    
    Args:
        image: This is the original Image
        masks: This is the mask for the image
        boxes: This is the predicted bounding box for the image
        labels: This is the predicted label for the image
        
    Returns:
        image: This is the image with the segmentation drawn on it
        
    """
    alpha =1
    beta = 0.6 # This is the alpha value for the color of the mask (transparency)
    gamma = 0
    
    for i in range(len(masks)):
       red_map = np.zeros_like(masks[i]).astype(np.uint8)
       green_map = np.zeros_like(masks[i]).astype(np.uint8)
       blue_map = np.zeros_like(masks[i]).astype(np.uint8)
       
       
       # Now we need to create the color map for the mask
       color = COLORS[random.randrange(0,len(COLORS))]
       
       red_map[masks[i] ==1],green_map[masks[i] ==1],blue_map[masks[i] ==1] = color
       
       seg_map = cv2.merge([red_map,green_map,blue_map])
       
       image = np.array(image)
       
       image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
       
       cv2.addWeighted(image,alpha,seg_map,beta,gamma,image)
       
       cv2.rectangle(image,boxes[i][0],boxes[i][1],color = color ,thickness = 2)
        
       cv2.putText(image,labels[i],(boxes[i][0][0],boxes[i][0][1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness = 2,lineType = cv2.LINE_AA)
       
       
    return image
       
       
def mask_from_image_user_select_label(masks,labels):
    """
    @brief: This function will return the mask for the selected label
    
    Args:
        image: This is the image which will be used for segmentation
        label: This is the label for the image
        model: Mask R-CNN model
        threshold: Threshold for the model ( pre-defined score below which we will ignore the mask)
        
    Returns:
        mask: This is the mask for the selected label
        
    """
    
    # Now we need to get the mask for the selected label

    pass
def show_mask_with_img(image,mask,label):
    image = np.asarray(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    
    cv2.addText(image,label,(0,0),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("image",image)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   

def detect_voice_to_text(flag):
    
    while (flag == True):
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Please say your object name which is to be removed from the image and say stop at the end")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        try:
           
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
        except:
            print("Sorry could not recognize your voice")
    return text
    




def merge_masks(masks,labels,user_selection):
    mask_ind = []
    # Merge masks from the user selected labels 
    for i in range(0,len(masks)):
        if labels[i] in user_selection:
            mask_ind.append(i)
        
    mask_ind.sort()  
 
    mask = np.zeros_like(masks[0]*255)
    for i in mask_ind:
        mask += np.array(masks[i]*255)
    mask = np.array(mask,dtype = np.uint8)
    return mask


def visualize_mask(image,masks,label):
    
     
    alpha =1
    beta = 0.6 # This is the alpha value for the color of the mask (transparency)
    gamma = 0
    
    for i in range(len(masks)):
       red_map = np.zeros_like(masks[i]).astype(np.uint8)
       green_map = np.zeros_like(masks[i]).astype(np.uint8)
       blue_map = np.zeros_like(masks[i]).astype(np.uint8)
       
        
       # Now we need to create the color map for the mask
       color = COLORS[random.randrange(0,len(COLORS))]
       
       red_map[masks[i] ==1],green_map[masks[i] ==1],blue_map[masks[i] ==1] = color
       
       seg_map = cv2.merge([red_map,green_map,blue_map])
       
       image = np.array(image)
       
       image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
       
       cv2.addWeighted(image,alpha,seg_map,beta,gamma,image)
        
       cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),color = color ,thickness = 2)
      
       cv2.putText(image,label,(0,0),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness = 2,lineType = cv2.LINE_AA)       
       
       
       return image
    
    
def visualize_mask_2(image,masks,labels,selected_label):
    """
    @brief: This function will visualize the mask
    
    Args:
        image: This is the image which will be used for segmentation
        mask: This is the mask for the image
        label: This is the label for the image
        
    Returns:
        image: This is the image with the mask drawn on it
        
    """
    image = np.array(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for i in range(0,len(masks)):
        if labels[i] == selected_label:
            mask = masks[i]
            red_map = np.zeros_like(mask).astype(np.uint8)
            green_map = np.zeros_like(mask).astype(np.uint8)
            blue_map = np.zeros_like(mask).astype(np.uint8)
            color = COLORS[random.randrange(0,len(COLORS))]
            red_map[mask ==1],green_map[mask ==1],blue_map[mask ==1] = (255,0,0)
            seg_map = cv2.merge([red_map,green_map,blue_map])
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.addWeighted(image,1,seg_map,0.6,0,image)
            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),color = color ,thickness = 2)
            cv2.putText(image,labels[i],(0,image.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness = 2,lineType = cv2.LINE_AA)
            
    return image
    

    
if __name__=='__main__':
    detect_voice_to_text(flag=True)


    
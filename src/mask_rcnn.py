import torch
import torchvision
import cv2
import argparse

from PIL import Image
from utils import *
from torchvision.transforms import transforms as transforms



parser = argparse.ArgumentParser(description='Mask R-CNN for Instance Segmentation')
parser.add_argument('--image_path','--input',required=True,help='Path to the image')
parser.add_argument('-t','--threshold',default=0.965,type = float,help='Threshold for the model')
args = vars(parser.parse_args())


model =torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,progress = True ,num_classes=91)


# Now we set the computation device to CPU or GPU if cuda is available

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Now we load the model to the device and evaluate the model
model.to(device).eval()

# Now we transform the image to the required size(tensor)
transform = transforms.Compose([transforms.ToTensor()])

image_path = args['image_path']
image = Image.open(image_path).convert('RGB')

original_img = image.copy()

# We transform the image 
image = transform(image)

# We add the batch dimension to the image using unsqueeze
image = image.unsqueeze(0).to(device)

masks,boxes,labels = get_outputs(image,model,args['threshold'])



# res = np.array(masks[0]*255,dtype=np.uint8)
# print(res)


# res = np.array(masks[0]*255,dtype = np.uint8) + np.array(masks[1]*255,dtype = np.uint8)


# res = merge_masks(masks,labels,['elephant-0', 'elephant-1','elephant-2'])

result = merge_masks(masks,labels,labels)
result_img =  result * 255.0
# result = draw_segmentation(original_img,masks,boxes,labels)
# res = visualize_mask(original_img,masks,'elephant-1')
# res2 =visualize_mask_2(original_img,masks,labels,selected_label=['elephant-1','elephant-4'])
# mask,label = mask_from_image_user_select_label(masks,'person')
result_img = np.dstack((result_img,result_img,result_img))
# res = cv2.bitwise_or(original_img,result_img)
cv2.imshow("The",result_img)
# show_mask_with_img(original_img,masks[a],'person')
# Now we show the image
# cv2.imshow('Segmented Image',result)
cv2.imwrite("/home/raul/Documents/Capstone_673/GMM/output/segmented_image.jpg",result)
# cv2.imshow("Mask",res)

cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow('Segmented Image 2',res)
# cv2.imshow('Segmented Image 3',res2)
# cv2.waitKey(0)

# Now we save the image
# save_path = f"../output/{args['input'].split('/')[-1].split('.')[0]}.png"
# cv2.imwrite(save_path,result)

    


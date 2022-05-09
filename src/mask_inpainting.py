########################################################################################################################
#                                  INPAINTING THE VIDEO                                                                #
########################################################################################################################


"""
@brief: This file contains the functions to implement the mask inpainting algorithm.

"""

import cv2
import numpy as np
from yaml import ValueToken
import neuralgym as ng
import traceback
from helper import *
import logging
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL # Set the logging level to only log errors.
from inpaint_ops import *
from inpaint_model import *


def inpaint_image(img_path,mask_path,out_path,chk_dir):
    """
    @brief: This function implements the mask inpainting algorithm on a single image.
    
    Args:
        img_path: The path to the image to be inpainted.
        mask_path: The path to the mask of the image.
        out_path: The path to the output image.
        chk_dir: The path to the checkpoint directory.

    Returns:
        It writes the output image to the output path.
    
    """
    
    img = cv2.imread(img_path) # Read the image.
    mask = cv2.imread(mask_path) # Read the mask.
    
    with HiddenPrints():
        flag = ng.Config('inpaint.yml') # Read the configuration file.
    model = InpaintCAModel() # Create the model.
    
    assert img.shape == mask.shape # Check if the image and mask have the same shape.

    h, w, _ = img.shape # Get the height, width and number of channels of the image.
    
    grid_size = 8 # Set the grid size. This is the size of the image that will be processed at a time.
    
    image = img[:h//grid_size*grid_size, :w//grid_size*grid_size, :] # Crop the image to the grid size.
    mask = mask[:h//grid_size*grid_size, :w//grid_size*grid_size, :] # Crop the mask to the grid size.
    
    # Now we have the image and the mask cropped to the grid size.
    
    # We need to expand the image and mask to the same size as the input image.
    
    image = np.expand_dims(image, 0) # Expand the image to a batch of size 1.
    mask = np.expand_dims(mask, 0) # Expand the mask to a batch of size 1.
    img_input = np.concatenate([image, mask], axis=2) # Concatenate the image and mask.
    
    tf.reset_default_graph() # Reset the default graph. This is necessary to avoid errors.
    
    session_config = tf.ConfigProto() # Create a session configuration.
    session_config.gpu_options.allow_growth = True # Allow the GPU to grow if needed.
    with tf.Session(config=session_config) as sess:
        img_input = tf.constant(img_input, dtype=tf.float32) # Create a tensorflow constant from the image and mask.
        output = model.build_server_graph(flag, img_input,reuse = False) # Build the graph.
        output = (output +  1) * 127.5 # Normalize the output. This is necessary because the output is in the range [-1, 1].
        output = tf.reverse(output, [-1]) # Reverse the output. 
        output = tf.saturate_cast(output, tf.uint8) # Cast the output to uint8. 
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) # Get the global variables. 
        
        # Load the variables from the checkpoint. 
        
        var_list = []   # Create a list to store the variables.
        
        for var in variables: # For each variable. 
            variable_name = var.name # Get the variable name.
            from_name = variable_name # Set the from name to the variable name.
            var_value = tf.contrib.framework.load_variable(chk_dir, from_name) # Load the variable from the checkpoint.
            var_list.append(tf.assign(var, var_value)) # Assign the variable to the list.
        sess.run(var_list) # Assign the variables. 
        
        result = sess.run(output) # Run the graph. 
        cv2.write(out_path, result[0][:,:,::-1]) # Write the output to the output path.
        
def inpaint_video(video_path):  
    """
    @brief: This function implements the mask inpainting algorithm on a video.
    
    Args:
        video_path: The path to the video to be inpainted.
        
    Returns:
        It writes the output video to the output path.
        
    """ 
    chk_dir  = '/checkpoints/release_places2_256' # Set the checkpoint directory.
    output_dir = "/output/" + video_path.split(".")[0].split('/')[-1] + "/" # Set the output directory.
    
    k = 0 # Set the counter to 0.
    while True:
        input_path = output_dir + video_path.split(".")[0].split('/')[-1] + "_" + str(k) + ".jpg" # Set the input path.
        mask_path = output_dir + video_path.split(".")[0].split('/')[-1] + "_" + str(k) + "_mask.jpg" # Set the mask path.
        out_path = output_dir + video_path.split(".")[0].split('/')[-1] + "_" + str(k) + "_inpainted.jpg" # Set the output path.
        k += 1 # Increment the counter.
        
        try:
            inpaint_image(input_path,mask_path,out_path,chk_dir) # Inpaint the image.
        except Exception as e:
            
            track = traceback.format_exc() # Get the traceback.
            if not track == traceback.format_exc():
                print("Traceback:",track) # Print the traceback.
            break
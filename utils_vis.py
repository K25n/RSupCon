"""
Modified Version of Convolutional Neural Network Visualizations
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/generate_class_specific_samples.py

"""

import torch
from PIL import Image, ImageFilter
from torch.autograd import Variable
import numpy as np
import os
from torch.optim import SGD
import copy
import torchvision.transforms.functional as F

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")
    # Resize image
    if resize_im:
        pil_im.thumbnail((32, 32))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
    #     im_as_arr[channel] -= mean[channel]
    #     im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable

    im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    # for c in range(3):
    #     recreated_im[c] /= reverse_std[c]
    #     recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
        im = im.resize((64, 64))
    im.save(path)

def rgb_img(opt):

    if opt.color == "random":
        # 랜덤한 RGB 값 생성
        red = np.random.randint(0, 255, size=(32, 32))
        green = np.random.randint(0, 255, size=(32, 32))
        blue = np.random.randint(0, 255, size=(32, 32))
    elif opt.color == "black":
        red = np.zeros((32, 32))
        green = np.zeros((32, 32))
        blue = np.zeros((32, 32))
    elif opt.color == "white":
        red = 255 * np.ones((32, 32))
        green = 255 * np.ones((32, 32))
        blue = 255 * np.ones((32, 32))
    elif opt.color == "blue":
        red = np.zeros((32, 32))
        green = np.zeros((32, 32))
        blue = 255 * np.ones((32, 32))
    elif opt.color == "green":
        red = np.zeros((32, 32))
        green = 255 * np.ones((32, 32))
        blue = np.zeros((32, 32))
    elif opt.color == "red":
        red = 255 * np.ones((32, 32))
        green = np.zeros((32, 32))
        blue = np.zeros((32, 32))
    else:
        raise ValueError("Invalid color option: {}".format(opt.color))

    return red, green, blue

class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class,name,rgb_set=False,color="random"):
        
        cifar = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
        ]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.name= name
        self.categories = cifar[target_class]
        self.target_class_name = cifar[target_class]
        self.color = color
        # Generate a random image
        if rgb_set:
            red, green, blue = rgb_img(self)
            self.created_image = np.stack([red, green, blue],axis=-1)
            self.categories += '_'+color
        else:
            self.created_image = np.uint8(np.random.uniform(0, 255, (32, 32, 3)))

        # Create the folder to export images if not exists
        if not os.path.exists(f'./vis/generated/{self.name}/class_{self.target_class_name}'):
            os.makedirs(f'./vis/generated/{self.name}/class_{self.target_class_name}')


    def generate(self):
        initial_learning_rate = 6


        for i in range(101):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            self.processed_image = self.processed_image.cuda()
            
            
            wd=0.0001
            
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate, weight_decay = wd)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())
            if i % 10 == 0:
                # Save image
                max_logit = output.max().item()
                im_path = f'./vis/generated/{self.name}/class_{self.target_class_name}/c{self.target_class}_specific_iteration_'+self.categories+f'_{int(max_logit)}_'+str(i)+'.jpg'
                #im_path = '../generated/c_specific_iteration_r_'+type(self.model).__name__+'_'+self.categories+'_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
        return self.processed_image
    

def max_act_img(trainloader,model,name): 
    class_logits_and_images = {}
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
        ]

    model.eval()
    if not os.path.exists(f'./vis/generated/{name}/'):
            os.makedirs(f'./vis/generated/{name}/')

    with torch.no_grad():
        for class_id in range(10):
            max_logit = float("-inf")
            max_image = None
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = inputs.cuda(), labels.cuda()


                outputs = model(inputs)

                max_logit_in_batch, max_index = torch.max(outputs[:, class_id], dim=0)
                max_logit_in_batch = max_logit_in_batch.item()
                max_image = inputs[max_index].cpu()

                if max_logit_in_batch > max_logit:
                    max_logit = max_logit_in_batch


            max_image = F.to_pil_image(max_image)
            image_filename = f"./vis/generated/{name}/class_{class_id}_highest_logit_{max_logit}_image64.png"
            max_image = max_image.resize((64, 64))
            max_image.save(image_filename)

            class_logits_and_images[class_id] = {
                'logit': max_logit,
                'image_filename': image_filename
            }


import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

IMG_SIZE = 512

def process_image(image_file, device):
    img = Image.open(image_file).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3) # force B&W
    ])
    
    img_resized = transform(img)
    img_np = np.array(img_resized)
    
    img_lab = rgb2lab(img_np).astype('float32')
    L = img_lab[:, :, 0] / 100.0  # normalize L (0-1)
    
    # create tensor: (batch, channel, h, w) -> (1, 1, 512, 512)
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    
    return img_resized, L_tensor

def postprocess_output(L_tensor, ab_tensor): # combines input L channel with predicted ab channels
    L = L_tensor.squeeze().cpu().detach().numpy() * 100.0
    ab = ab_tensor.squeeze().cpu().detach().numpy() * 128.0
    
    h, w = L.shape
    lab_out = np.zeros((h, w, 3))
    
    # fill channels
    lab_out[:, :, 0] = L
    lab_out[:, :, 1:] = ab.transpose(1, 2, 0)
    
    rgb_out = lab2rgb(lab_out)
    return (rgb_out * 255).astype(np.uint8) # lab2rgb is float
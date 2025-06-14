import os
import sys
import numpy as np
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--image_path', type=str, default='/home/Github/SCI/2015_01928.png',
                    help='path to the input image')
parser.add_argument('--save_path', type=str, default='./results/medium', help='location to save the processed image')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='path to the model weights')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

def save_image(tensor, path):
    # Convert tensor to numpy array
    image_numpy = tensor[0].cpu().float().numpy()
    # Transpose dimensions
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    # Clip pixel values and convert to uint8
    image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    # Convert numpy array to PIL image
    im = Image.fromarray(image_numpy)
    # Apply horizontal flip
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # Rotate image 90 degrees clockwise
    im = im.rotate(90, expand=True)
    # Save the image
    im.save(path, 'png')

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()
    model.eval()

    # Load and preprocess input image
    input_image = Image.open(args.image_path).convert('RGB')
    input_tensor = torch.unsqueeze(torch.transpose(torch.tensor(np.array(input_image), dtype=torch.float32) / 255.0, 0, 2), 0).cuda()
    input_var = Variable(input_tensor, volatile=True)

    # Forward pass
    with torch.no_grad():
        _, result = model(input_var)

    # Save processed image
    save_image(result, os.path.join(save_path, 'processed_image.png'))

if __name__ == '__main__':
    main()

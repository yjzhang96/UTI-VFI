import torch 
import numpy as np 
import os
from PIL import Image

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().detach().numpy()
    image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0))+0.5,0,1)  * 255.0
    return image_numpy.astype(imtype)

def load_image(filename, trans_list=None, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    if trans_list:
        img = trans_list(img)
    img = img.unsqueeze(0)
    return img


def save_image(image_numpy, image_path):
    image_pil = None
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_train_sample(args,epoch, results):
    save_dir = os.path.join(args.checkpoints, args.model_name, 'sample')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for label, image_numpy in results.items():
        save_path = os.path.join(save_dir, 'epoch%.3d_%s.png'%(epoch,label))
        save_image(image_numpy, save_path)

def save_test_images(args, save_dir, results, image_path):
    blur_inter = args.blur_interval
    B1_path = image_path['B1_path']
    B2_path = image_path['B2_path']
    # import ipdb; ipdb.set_trace()
    # save_dir = os.path.join(args.result_dir,args.model_name) 
    others, B1_name = os.path.split(B1_path)
    B2_name = os.path.split(B2_path)[-1]
    video_name = os.path.split(others)[-1]

    video_dir = os.path.join(save_dir, video_name)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    
    frame_index_B1 = int(B1_name.split('.')[0])
    frame_index_B2 = int(B2_name.split('.')[0])

    save_B1S_path = os.path.join(video_dir,"%06d.png"%(frame_index_B1-blur_inter//2))
    save_B1E_path = os.path.join(video_dir,"%06d.png"%(frame_index_B1+blur_inter//2))
    save_B2S_path = os.path.join(video_dir,"%06d.png"%(frame_index_B2-blur_inter//2))
    save_B2E_path = os.path.join(video_dir,"%06d.png"%(frame_index_B2+blur_inter//2))
    
    save_image(results['B1_S'],save_B1S_path)
    save_image(results['B1_E'],save_B1E_path)
    save_image(results['B2_S'],save_B2S_path)
    save_image(results['B2_E'],save_B2E_path)


def save_inter_images(image_tensor, image_path, trans_list=None):
    if trans_list==None:
        img = tensor2im(image_tensor)
    else:
        img = trans_list(image_tensor)
        img = tensor2im(img)
    save_image(img, image_path)
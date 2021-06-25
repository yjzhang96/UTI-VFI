import argparse
import time
import os
import ipdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from shutil import rmtree
from models import model_SEframe_stage2, model_refineS2
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument("--blurry_videos", type=str, default='/home/yjz/datasets/LFR_gopro_91', help='directory path to  blurry videos')
parser.add_argument("--sharp_videos", type=str, default='/home/yjz/datasets/GOPRO', help='If training, directory path to  sharp videos')
parser.add_argument("--checkpoints", type=str,default='./checkpoints', help='path to pretrained model, for resume training or test')
parser.add_argument("--result_dir", type=str,default='./results', help='result restore path')
parser.add_argument("--model_name", type=str, default='SEFrame' ,help='model name, for resume/save training or test')
parser.add_argument("--resume_train", action='store_true', help='resume training')
parser.add_argument("--which_epoch", type=str, default='latest' ,help='which epoch to load? set to latest to use latest cached model')
parser.add_argument("--dataset_mode", type=str, default='mix' ,help='2frames / 4frames')
parser.add_argument("--crop_size_X", type=int, default=256)
parser.add_argument("--crop_size_Y", type=int, default=256)
parser.add_argument("--gpu", type=str, default="0",help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--test_batch_size", type=int, default=1,help='validation batch szie')
parser.add_argument("--train", type=bool, default= False,help='training')
parser.add_argument("--save_dir", type=str,default='', help='result restore path')
parser.add_argument("--inter_root", type=str,default='', help='result restore path')
parser.add_argument("--deblur", type=int, default=0,help='run deblur')
parser.add_argument("--inter", type=int, default=0,help='run inter')

# interpolate parameters
parser.add_argument("--inter_frames", type=int, default=10,help='total number of frames to be interpolated')
parser.add_argument("--blur_interval", type=int, default=9,help='number of frames compound blurry frame')
parser.add_argument("--test_type", type=str, required=True ,help='validation / real_world')

args = parser.parse_args()

if "9" in args.blurry_videos:
    args.blur_interval = 9
elif "7" in args.blurry_videos:
    args.blur_interval = 7
elif "5" in args.blurry_videos:
    args.blur_interval = 5
else:
    raise('incorrect blur interval')
    
### make saving dir
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if args.save_dir:
    image_save_dir = os.path.join(args.result_dir,args.save_dir) 
else:
    image_save_dir = os.path.join(args.result_dir,args.model_name) 

if not os.path.exists(image_save_dir):
    os.mkdir(image_save_dir)

deblur = args.deblur
normalize1 = transforms.Normalize([0.5,0.5,0.5], [1.0, 1.0, 1.0])
trans_list = transforms.Compose([transforms.ToTensor(), normalize1])



# testing deblur
if deblur:
    ### initialize model
    model = model_SEframe_stage2.SEframeNet(args)
    load_file = 'pretrain_models' + '/' + 'SEframe_net.pth'
    model.SE_deblur_net.load_state_dict(torch.load(load_file))
    print('--------load model %s success!-------'%load_file)

    
    blur_dir = args.blurry_videos
    videos = sorted(os.listdir(blur_dir))
    for video in videos:
        
        video_dir= os.path.join(blur_dir,video)
        frames = sorted(os.listdir(video_dir))
        start_time = time.time()
        print('--------testing begin----------')
        for i in range(len(frames)-3):
            # load blurry video frames
            I0 = utils.load_image(os.path.join(video_dir,frames[i]),trans_list)
            I1 = utils.load_image(os.path.join(video_dir,frames[i+1]),trans_list)
            I2 = utils.load_image(os.path.join(video_dir,frames[i+2]),trans_list)
            I3 = utils.load_image(os.path.join(video_dir,frames[i+3]),trans_list)
            
            # if index%2 == 0:
            batch_data = [I0,I1,I2,I3]
            model.set_test_input(batch_data)
            psnr = model.test()
            results = model.get_current_visuals()
            image_path = {'B1_path':os.path.join(video_dir,frames[i+1]),'B2_path':os.path.join(video_dir,frames[i+2])}
            utils.save_test_images(args, image_save_dir, results, image_path)
            print('processing %s'%(image_path['B1_path']))


# testing interpolation
interpolate = args.inter
save_dir = '%s/%s'%(args.result_dir, args.save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)

def load_my_state_dict(model, parameter_dict):
    own_state = model.state_dict()
    for k, v in parameter_dict.items():
        own_state[k].copy_(v.data)

if interpolate:
    video_inter_root= args.inter_root
    if not os.path.exists(video_inter_root):
        raise ImportError('interpolate root directory not exists')
    from models.inter_models import UTI_model
    esti_model = UTI_model.UTI_esti().cuda()
    inter_model = UTI_model.UTI_interp().cuda()
    
    inter_model = nn.DataParallel(inter_model)
    print('--------------load network--------------')
    dict = torch.load('./utils/model.ckpt')
    # for k, v in dict['model_state_dict'].items():
    #     print(k)
    load_my_state_dict(inter_model,dict['model_state_dict'])

    videos = sorted(os.listdir(video_inter_root))
    for video in videos:
        src_frames = sorted(os.listdir(os.path.join(video_inter_root,video)))
        print('start processing video %s'%video)
        extraction_dir = '.extraction'
        if not os.path.exists(extraction_dir):
            os.mkdir(extraction_dir)
        print('making tmp extraction dir %s'%extraction_dir)
        if not os.path.exists(os.path.join(save_dir,video)):
            os.mkdir(os.path.join(save_dir,video))
            print('make dir %s'%(os.path.join(save_dir,video)))
        
        for i in range(len(src_frames)):
            os.system('cp {} {}'.format(os.path.join(video_inter_root,video,src_frames[i]), os.path.join(extraction_dir,src_frames[i]))) 
        frames = sorted(os.listdir(extraction_dir))

        if args.test_type == 'validation':
            for i in range(len(frames)-3):

                I0 = utils.load_image(os.path.join(extraction_dir,frames[i]),trans_list).cuda()
                I1 = utils.load_image(os.path.join(extraction_dir,frames[i+1]),trans_list).cuda()
                I2 = utils.load_image(os.path.join(extraction_dir,frames[i+2]),trans_list).cuda()
                I3 = utils.load_image(os.path.join(extraction_dir,frames[i+3]),trans_list).cuda()

                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i]),os.path.join(save_dir,video,frames[i])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+1]),os.path.join(save_dir,video,frames[i+1])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+2]),os.path.join(save_dir,video,frames[i+2])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+3]),os.path.join(save_dir,video,frames[i+3])))

                index_0 = int(frames[i].split('.')[0])
                index_1 = int(frames[i+1].split('.')[0])
                index_2 = int(frames[i+2].split('.')[0])
                
                inter_n = index_2 -index_1  # This inter- and intra- frame numbers are only calculated for indexing the output,
                intra_n = index_1 -index_0  # which means they are not used for calculate temporal ratio.            

                with torch.no_grad():
                    flow , lambda_t = esti_model(I0,I1,I2,I3)
                    for tt in range(inter_n-1):
                        # tt is the index of inter frame
                        time_tt = (tt+1)/inter_n
                        
                        output = inter_model(I1, I2, flow, lambda_t, time_tt)
                        save_path = os.path.join(save_dir, video, '%06d.png'%(tt+1+index_1))
                        utils.save_inter_images(output, save_path)
                        print('generating image %s'%save_path)
        elif args.test_type == 'real_world':
            for i in range(len(frames)-3):

                I0 = utils.load_image(os.path.join(extraction_dir,frames[i]),trans_list).cuda()
                I1 = utils.load_image(os.path.join(extraction_dir,frames[i+1]),trans_list).cuda()
                I2 = utils.load_image(os.path.join(extraction_dir,frames[i+2]),trans_list).cuda()
                I3 = utils.load_image(os.path.join(extraction_dir,frames[i+3]),trans_list).cuda()
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i]),os.path.join(save_dir,video,frames[i])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+1]),os.path.join(save_dir,video,frames[i+1])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+2]),os.path.join(save_dir,video,frames[i+2])))
                os.system('cp %s %s'%(os.path.join(extraction_dir,frames[i+3]),os.path.join(save_dir,video,frames[i+3])))
                index_1 = int(frames[i+1].split('.')[0])

                with torch.no_grad():
                    flow , lambda_t = esti_model(I0,I1,I2,I3)

                    lambda_ave = (lambda_t['forward'].mean() + lambda_t['backward'].mean())/2
                    inter_n = int(torch.round((lambda_ave * args.inter_frames - 1 + lambda_ave) / (lambda_ave + 1)))
                    intra_n = args.inter_frames - 2 - inter_n
                    print('estimate time interval:%.02fs'%(lambda_ave/(lambda_ave+1)).item())
                    print("estimate inter-frame number:%d"%inter_n)
                    
                    for tt in range(inter_n):
                        # tt is the index of inter frame
                        time_tt = (tt+1)/inter_n
                        
                        output = inter_model(I1, I2, flow, lambda_t, time_tt)

                        save_path = os.path.join(save_dir, video, '%06d_%02d.png'%(index_1,tt+1))
                        utils.save_inter_images(output, save_path)
                        print('generating image %s'%save_path)

        print('finishing processing video %s'%video)
        rmtree(extraction_dir)
        print('remove tmp extraction dir:%s'%extraction_dir)
            

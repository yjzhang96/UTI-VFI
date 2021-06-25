import argparse
import time
import os
import ipdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from data import dataloader_4frame

from models import model_SEframe_stage1
from models import model_SEframe_stage2
from utils import utils
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--blurry_videos_root", type=str, default='/home/yjz/hdd2/datasets/', help='directory path to  blurry videos')
parser.add_argument("--blurry_videos_dir", type=str, default='LFR_gopro_55,LFR_gopro_73,LFR_gopro_91', help='directory path to  blurry videos')
parser.add_argument("--sharp_videos", type=str, default='/home/yjz/hdd2/datasets/GOPRO_GT_720p', help='If training, directory path to  sharp videos')
parser.add_argument("--checkpoints", type=str,default='./checkpoints', help='path to pretrained model, for resume training or test')
parser.add_argument("--model_name", type=str, default='SEFrame' ,help='model name, for resume/save training or test')
parser.add_argument("--train_stage", type=int, default=1 ,help='which training stage, 1 = first residual/2 = second residual')
parser.add_argument("--resume_train", action='store_true', help='resume training')
parser.add_argument("--which_epoch", type=str, default='latest' ,help='which epoch to load? set to latest to use latest cached model')
parser.add_argument("--dataset_mode", type=str, default='4frames' ,help='4frames')
parser.add_argument("--start_epoch", type=int, default=0 ,help='which epoch to start training')
parser.add_argument("--save_epoch", type=int, default=10, help='save model frequency')
parser.add_argument("--display_freq", type=int, default=100, help='display frequency every xx steps')
parser.add_argument("--val_freq", type=int, default=5, help='validation frequency')
parser.add_argument("--pwc_path", type=str, default='./utils/network-default.pytorch' , help='pretrained PWC model')
parser.add_argument("--crop_size_X", type=int, default=256)
parser.add_argument("--crop_size_Y", type=int, default=256)
parser.add_argument("--gpu", type=str, default="0",help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--blur_interval", type=str, default='5,7,9',help='number of frames compound blurry frame')

# training cd  
parser.add_argument("--train", type=bool, default=True,help='training')
parser.add_argument("--lr", type=float, default=0.0001,help='learning rate')
parser.add_argument("--beta1", type=float, default=0.9,help='momentum term of Adam optimizer')
parser.add_argument("--epoch", type=int, default=50,help='total epochs for training')
parser.add_argument("--batch_size", type=int, default=4,help='batch szie')
parser.add_argument("--val_batch_size", type=int, default=1,help='validation batch szie')


args = parser.parse_args()

### make saving dir
if not os.path.exists(args.checkpoints):
    os.mkdir(args.checkpoints)
model_save_dir = os.path.join(args.checkpoints,args.model_name) 
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)



### initialize model
if args.train_stage == 1:
    Model = model_SEframe_stage1
elif args.train_stage == 2:
    Model = model_SEframe_stage2
    
model = Model.SEframeNet(args)

### load datasets
# dataset mode = 4frames or 2frames
if args.dataset_mode == '4frames':
    train_dataset = dataloader_4frame.LFRvideo(args, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = True)
    val_dataset = dataloader_4frame.LFRvideo(args, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.val_batch_size,
                                    shuffle = True)
else:
    raise ValueError("Dataset mode %s not recognized"%args.dataset_mode)
print(train_dataset)
print(val_dataset)

###Create transform to display image from tensor


###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



### Initialization
if args.resume_train:
    model.load(args)
    if args.which_epoch != 'latest':
        args.start_epoch = int(args.which_epoch)
    elif args.which_epoch == 'latest':
        assert args.start_epoch != 0
    model.get_current_lr_from_epoch(args.lr,args.start_epoch, args.epoch)
else:
    os.system('rm %s/%s/psnr_log.txt'%(args.checkpoints, args.model_name))
    os.system('rm %s/%s/event*'%(args.checkpoints, args.model_name))

# tensorboard writter
writer = SummaryWriter(model_save_dir)

def display_loss(loss,epoch,tot_epoch,step,step_per_epoch):
    loss_writer = ""
    for key, value in loss.items():
        loss_writer += "%s:%.4f\t"%(key,value)
    print("epoch[%d/%d],step[%d/%d]:%s"%(epoch,tot_epoch,step,step_per_epoch,loss_writer))

# validation
def validation(epoch):
    t_psnr = 0
    cnt = 0
    start_time = time.time()
    print('--------validation begin----------')
    for index, batch_data in enumerate(val_dataloader):
        model.set_input(batch_data)
        psnr = model.test(validation=True)
        t_psnr += psnr
        cnt += 1
        if index > 400:
            break
    message = 'validation epoch %s PSNR: %.2f'%(epoch, t_psnr/cnt)
    print(message)
    print('using time %.3f'%(time.time()-start_time))
    log_name = os.path.join(args.checkpoints,args.model_name,'psnr_log.txt')   
    with open(log_name,'a') as log:
        log.write(message+'\n')
    return t_psnr/cnt 

# training
# val_psnr = validation(args.start_epoch)
for epoch in range(args.start_epoch, args.epoch):
    epoch_start_time = time.time()
    step_per_epoch = len(train_dataloader)    
    for step, batch_data in enumerate(train_dataloader):
        model.set_input(batch_data)      
        model.optimize() # step(), backward()
        if step%5 == 0:
            loss = model.get_loss()
            display_loss(loss,epoch,args.epoch,step,step_per_epoch)
            writer.add_scalar('loss',loss['total_loss'],step)
        if step%args.display_freq == 0:
            #print a sample result in checkpoints/model_name/samples
            results = model.get_current_visuals()
            utils.save_train_sample(args, epoch, results)

    # schedule learning rate
    model.schedule_lr(epoch,args.epoch)
    model.save('latest')
    print('End of epoch [%d/%d] \t Time Taken: %d sec' % (epoch, args.epoch, time.time() - epoch_start_time))

    if epoch%args.save_epoch == 0:
        model.save(epoch)

    if epoch%args.val_freq == 0:
        val_psnr = validation(epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)



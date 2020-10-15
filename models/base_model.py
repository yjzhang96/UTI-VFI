import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from .networks import RDN_residual_deblur
from collections import OrderedDict
from .PWCNetnew import PWCNet
from utils import utils

class SEframeNet():
    def __init__(self, args):
        self.opt = args
        if args.gpu:
            self.device = torch.device('cuda:{}'.format(args.gpu[0]))
        else:
            self.device = torch.device('cpu')

        ### initial model
        self.flow_net = PWCNet()
        # import ipdb; ipdb.set_trace()
        self.flow_net.load_state_dict(torch.load(args.pwc_path))
        self.flow_net.to(self.device)
        self.SE_deblur_net = RDN_residual_deblur()
        self.SE_deblur_net.to(self.device)
        ###Loss and Optimizer
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
        
        params = self.SE_deblur_net.parameters()
        if args.train:
            self.optimizer = optim.Adam(params, lr=args.lr, betas=(0.9,0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9,last_epoch=args.start_epoch)


    def set_input(self,batch_data):
        self.input_B1 = batch_data['B1'].to(self.device)
        self.input_B2 = batch_data['B2'].to(self.device)
        self.input_B1_S = batch_data['B1_S'].to(self.device)
        self.input_B1_E = batch_data['B1_E'].to(self.device)
        self.input_B2_S = batch_data['B2_S'].to(self.device)
        self.input_B2_E = batch_data['B2_E'].to(self.device)
        self.B1_path = batch_data['B1_path']
        self.B2_path = batch_data['B2_path']

    def forward(self):
        # calculate flow 
        flow_1_2 = self.flow_net(self.input_B1,self.input_B2)
        # get estimated start end frame
        out_B1_S,out_B1_E,out_B2_S,out_B2_E = self.SE_deblur_net(self.input_B1,self.input_B2,flow_1_2)
        
        
        # loss 
        loss_B1_S = self.L1_loss(self.input_B1_S,out_B1_S)
        loss_B1_E = self.L1_loss(self.input_B1_E,out_B1_E)
        loss_B2_S = self.L1_loss(self.input_B2_S,out_B2_S)
        loss_B2_E = self.L1_loss(self.input_B2_E,out_B2_E)
        self.output = {'B1_S':out_B1_S,'B1_E':out_B1_E,'B2_S':out_B2_S,'B2_E':out_B2_E}
        self.tot_loss = loss_B1_E + loss_B1_S + loss_B2_S + loss_B2_E
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        self.optimizer.step()
    
    def get_loss(self):
        
        return OrderedDict([('total_loss', self.tot_loss.item())])
    
    def test(self, validation = False):
        with torch.no_grad():
            flow_1_2 = self.flow_net(self.input_B1,self.input_B2)
            # get estimated start end frame
            out_B1_S,out_B1_E,out_B2_S,out_B2_E = self.SE_deblur_net(self.input_B1,self.input_B2,flow_1_2)
            self.output = {'B1_S':out_B1_S,'B1_E':out_B1_E,'B2_S':out_B2_S,'B2_E':out_B2_E}
        # calculate PSNR
        if validation:
            psnr = 0
            MSE = self.L2_loss(self.input_B1_S,out_B1_S) 
            psnr += 10 * np.log10(1 / MSE.item())
            MSE = self.L2_loss(self.input_B1_S,out_B1_S) 
            psnr += 10 * np.log10(1 / MSE.item())
            MSE += self.L2_loss(self.input_B1_S,out_B1_S) 
            psnr += 10 * np.log10(1 / MSE.item())
            MSE += self.L2_loss(self.input_B1_S,out_B1_S) 
            psnr += 10 * np.log10(1 / MSE.item())
            return psnr/4


    def save(self,epoch):
        save_filename = 'SEframe_net_%s.pth'%epoch
        save_path = self.opt.checkpoints + '/' + self.opt.model_name + '/' + save_filename
        if len(self.opt.gpu)>1:
            torch.save(self.SE_deblur_net.cpu().module.state_dict(),save_path)
        else:
            torch.save(self.SE_deblur_net.cpu().state_dict(),save_path)
        if self.opt.gpu:
            self.SE_deblur_net.to(self.device)

    def load(self, args):
        load_path = os.path.join(args.checkpoints, args.model_name)
        load_file = load_path + '/' + 'SEframe_net_%s.pth'%args.which_epoch
        self.SE_deblur_net.load_state_dict(torch.load(load_file))
        print('--------load model %s success!-------'%load_file)


    def schedule_lr(self, epoch):
        # scheduler
        # print("current learning rate:%.7f"%self.scheduler.get_lr())
        # self.scheduler.step()

        lr = self.opt.lr
        self.get_current_lr_from_epoch(lr, epoch)

    def get_current_lr_from_epoch(self, lr, epoch):
        decrease_step = 5
        current_lr = lr * (0.9**(epoch//decrease_step))
        if epoch > 200:
            current_lr = 0.000001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        print("current learning rate:%.7f"%(current_lr))

    def get_current_visuals(self):
        input_B1 = utils.tensor2im(self.input_B1)
        input_B2 = utils.tensor2im(self.input_B2)
        output_B1_S = utils.tensor2im(self.output['B1_S'])
        output_B1_E = utils.tensor2im(self.output['B1_E'])
        output_B2_S = utils.tensor2im(self.output['B2_S'])
        output_B2_E = utils.tensor2im(self.output['B2_E'])
        return OrderedDict([('B1',input_B1),('B2',input_B2),('B1_S',output_B1_S),
                            ('B1_E',output_B1_E),('B2_S',output_B2_S),('B2_E',output_B2_E)])
    
    def get_image_path(self):
        return {'B1_path':self.B1_path,'B2_path':self.B2_path}

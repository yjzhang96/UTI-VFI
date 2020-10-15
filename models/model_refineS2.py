import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from .networks import weights_init, Deblur_2step
from collections import OrderedDict
from .PWCNetnew import PWCNet
from utils import utils
from models.inter_models.refine_S2 import RefineS2

class RefineNet():
    def __init__(self, args):
        self.opt = args
        if args.gpu:
            self.device = torch.device('cuda:{}'.format(args.gpu[0]))
        else:
            self.device = torch.device('cpu')

        ### initial model
        self.flow_net = PWCNet()
        self.flow_net.load_state_dict(torch.load(args.pwc_path))
        self.flow_net.to(self.device)

        self.Refine_net = RefineS2().to(self.device)

        ###Loss and Optimizer
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()

        params = self.Refine_net.parameters()
        if args.train:
            self.optimizer = optim.Adam(params, lr=args.lr, betas=(0.9,0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9,last_epoch=args.start_epoch)


    def set_input(self,batch_data):
        self.input_B0 = batch_data['B0'].to(self.device)
        self.input_B1 = batch_data['B1'].to(self.device)
        self.input_B2 = batch_data['B2'].to(self.device)
        self.input_B3 = batch_data['B3'].to(self.device)
        self.B1_path = batch_data['B1_path']
        self.B2_path = batch_data['B2_path']
        self.r = batch_data['r']

    def forward(self):
        # calculate flow 
        with torch.no_grad():
            flow_10 = self.flow_net(self.input_B1,self.input_B0).float().detach()
            flow_12 = self.flow_net(self.input_B1,self.input_B2).float().detach()
            flow_13 = self.flow_net(self.input_B1,self.input_B3).float().detach()
            # flow_23 = self.flow_net(self.input_B2,self.input_B3).float().detach()
        flow_23 = flow_13 - flow_12
        r = self.r.cuda().float()
        # import ipdb; ipdb.set_trace()
        r = r.view(-1,1,1,1)
        flow_gt = 2*flow_12/r + flow_10 
        # get estimated start end frame
        flow_23_refine = self.Refine_net(flow_10,flow_12, flow_23)
                
        # loss 
        self.tot_loss = self.L2_loss(flow_23_refine,flow_gt)


        # self.output = {'B1_S':out2[0],'B1_E':out2[1],'B2_S':out2[2],'B2_E':out2[3]}
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        self.optimizer.step()
    
    def get_loss(self):
        
        return OrderedDict([('total_loss', self.tot_loss.item()),
                            ])
    
    def test(self, validation = False):
        with torch.no_grad():
            flow_10 = self.flow_net(self.input_B1,self.input_B0).float().detach()
            flow_12 = self.flow_net(self.input_B1,self.input_B2).float().detach()
            # flow_23 = self.flow_net(self.input_B2,self.input_B3).float().detach()
            flow_13 = self.flow_net(self.input_B1,self.input_B3).float().detach()
            flow_23 = flow_13 - flow_12

            
            r = self.r.cuda().float()
            flow_gt = 2*flow_12/r + flow_10 
            # get estimated start end frame
            flow_23_refine = self.Refine_net(flow_10,flow_12, flow_23)

        # self.output = {'B1_S':choose_out[0],'B1_E':choose_out[1],'B2_S':choose_out[2],'B2_E':choose_out[3]}
        if validation:
            EPE = self.L2_loss(flow_23_refine,flow_gt)
            return EPE


    def save(self,epoch):
        save_filename = 'refine_net_%s.pth'%epoch
        save_path = self.opt.checkpoints + '/' + self.opt.model_name + '/' + save_filename
        if len(self.opt.gpu)>1:
            torch.save(self.Refine_net.cpu().module.state_dict(),save_path)
        else:
            torch.save(self.Refine_net.cpu().state_dict(),save_path)
        if self.opt.gpu:
            self.Refine_net.to(self.device)

    def load(self, args):
        load_path = os.path.join(args.checkpoints, args.model_name)
        load_file = load_path + '/' + 'refine_net_%s.pth'%args.which_epoch
        self.Refine_net.load_state_dict(torch.load(load_file))
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
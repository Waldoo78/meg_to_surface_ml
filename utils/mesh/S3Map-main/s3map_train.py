#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 01:13:04 2021
Modified on 04/12/2023 for lifespan atlas construction

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""

import argparse
import numpy as np
import glob
import torch
import time
import os
import math
from s3pipe.models.models import SUnet
from s3pipe.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from s3pipe.utils.utils import get_neighs_order, get_sphere_template
from s3pipe.surface.s3map import ResampledInnerSurf, computeMetrics_torch
from torch.utils.tensorboard import SummaryWriter

###############################################################################

longleaf = True

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='train s3map model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hemi', default=None, help="hemisphere, lh or rh", required=True)
    parser.add_argument('--smooth', default=10.0, help="weights for training the model")
    parser.add_argument('--n_vertex', help="number of vertices", required=True)
    args = parser.parse_args()
    hemi = args.hemi
    weight_smooth = float(args.smooth)
    n_vertex = int(args.n_vertex)
    print('\nhemi:', hemi)
    print('weight_smooth:', weight_smooth)
    print('n_vertex:', n_vertex)
   
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    dataset = 'HighRes'   # 'HighRes' for dataset BCP and HCP with vertices number between 140000 - 250000, 'LowRes' for dataset dHCP with vertices between 30000 to 140000
    
    deform_scale = 10.0
    if n_vertex == 10242:
        weight_distan = 1.0
        weight_area = 0.1
    elif n_vertex == 40962:
        weight_distan = 1.0
        weight_area = 0.1
    elif n_vertex == 163842:
        weight_distan = 1.0
        weight_area = 0.1
    else:
        print("error")
        
    batch_size = 1
    learning_rate = 0.0001
    
    if longleaf:
        os.system('rm -rf /proj/ganglilab/users/Fenqiang/S3Pipeline/log/'+dataset+'_'+ hemi+'_' + str(n_vertex))
        writer = SummaryWriter('/proj/ganglilab/users/Fenqiang/S3Pipeline/log/'+dataset+'_'+ hemi+'_' + str(n_vertex))
        if dataset == 'HighRes':
            if n_vertex == 10242:
                files = sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/BCP/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.RespInner.npy')) + \
                        sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/HCP*/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.RespInner.npy'))
            elif n_vertex == 40962:
                files = sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/BCP/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.10242moved.RespInner.npy')) + \
                        sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/HCP*/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.10242moved.RespInner.npy'))
            elif n_vertex == 163842:
                files = sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/BCP/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.10242moved.40962moved.RespInner.npy')) + \
                        sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/HCP*/*/*.'+ \
                                     hemi +'.InnerSurf.inflated.initSphe.10242moved.40962moved.RespInner.npy'))
            else:
                raise NotImplementedError('error')
        else:
            raise NotImplementedError('error')
                
    else:
        files = sorted(glob.glob('/work/users/f/e/fenqiang/fenqiang/LifeSpanDatasetFQProcessed/dHCP/*/*.'+ \
                         hemi +'.InnerSurf.initSphe.RespInner.163842.npy'))
    
    
    print('len(files):', len(files))
    train_files = files
    print('len(train_files):', len(train_files))
    
    train_dataset = ResampledInnerSurf(train_files, n_vertex)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    if n_vertex == 10242:
        level = 6
        n_res = 4   
    elif n_vertex == 40962:
        level = 7
        n_res = 4
    elif n_vertex == 163842:
        level = 8
        n_res = 5
    else:
        print("error")
    
    model = SUnet(in_ch=12, out_ch=2, level=level, n_res=n_res, rotated=0, complex_chs=32)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.to(device)
    if longleaf:
        model_file = '/proj/ganglilab/users/Fenqiang/S3Pipeline/pretrained_models/S3Map_'+ \
                        dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                            '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl'
      
    else:
        model_file = '/media/ychenp/fq/S3_pipeline/SphericalMapping/pretrained_models/S3Map_'+ \
                        dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                            '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl'
    if os.path.isfile(model_file):
        print('Loading pre-trained model:', model_file)
        model.load_state_dict(torch.load(model_file))        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #############################################################
    
    def get_learning_rate(epoch):
        limits = [3, 6, 10]
        lrs = [1, 0.5, 0.1, 0.05]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate
    
    
    neigh_orders = get_neighs_order(n_vertex)
    neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_orders[:, 0:6]), axis=1)
    template = get_sphere_template(n_vertex)
    fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
    grad_filter = torch.ones((7, 1), dtype=torch.float32, device=device)
    grad_filter[6] = -6    
    bi_inter = get_bi_inter(n_vertex, device)[0]
    En = getEn(n_vertex, device)[0]
    
    
    # dataiter = iter(train_dataloader)
    # inner_vert, file = dataiter.next()
    
    
    for epoch in range(8):
        lr = get_learning_rate(epoch)
        optimizer.param_groups[0]['lr'] = lr
        print("learning rate = {}".format(lr))
        
        for batch_idx, (inner_vert, file) in enumerate(train_dataloader):
            model.train()
            file = file[0]
            inner_vert = inner_vert.squeeze().to(device)  
            inner_dist, inner_area, _ = computeMetrics_torch(inner_vert, neigh_sorted_orders, device)
    
            # feat = inner_dist    # 2023.6.21 use only metric feature        
            feat = torch.cat((inner_dist, inner_area), 1)  # features
            feat = feat.permute(1,0).unsqueeze(0)
            
            deform_2d = model(feat) / deform_scale
            deform_2d = deform_2d[0,:].permute(1,0)
            deform_3d = convert2DTo3D(deform_2d, En, device)
            # torch.linalg.norm(deform_3d, dim=1).mean()
            # deform_3d = deform_ratio.unsqueeze(2).repeat(1,1,3) * orig_sphere_vector_CtoN
            # deform_3d = torch.sum(deform_3d, dim=1)
       
            # moved_sphere_loc = fixed_xyz + deform_3d 
            # moved_sphere_loc = moved_sphere_loc / torch.linalg.norm(moved_sphere_loc, dim=1, keepdim=True).repeat(1,3)
                
            # diffeomorphic implementation
            velocity_3d = deform_3d/math.pow(2, 6)
            moved_sphere_loc = diffeomorp_torch(fixed_xyz, velocity_3d, 
                                              num_composition=6, bi=True, 
                                              bi_inter=bi_inter, 
                                              device=device)
            
            moved_dist, moved_area, _ = computeMetrics_torch(moved_sphere_loc, neigh_sorted_orders, device)
            
            
            loss_dist = torch.mean(torch.square((moved_dist - inner_dist) / (inner_dist+1e-12))) # 2023.6.13 change abs -> square
            loss_area = torch.mean(torch.square((moved_area - inner_area) / (inner_area+1e-12))) # 2023.6.21 remove area loss 
            
            loss_smooth = torch.abs(torch.mm(deform_3d[:,0][neigh_orders], grad_filter)) + \
                          torch.abs(torch.mm(deform_3d[:,1][neigh_orders], grad_filter)) + \
                          torch.abs(torch.mm(deform_3d[:,2][neigh_orders], grad_filter))
            loss_smooth = torch.mean(loss_smooth)
            
            loss = weight_distan * loss_dist + \
                   weight_smooth * loss_smooth + \
                   weight_area * loss_area 
                    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("[Epoch {}/{}] [loss_dist: {:5.4f}] [loss_area: {:5.4f}] [loss_smooth: {:5.4f}]".format(epoch, 
                                            batch_idx, loss_dist,loss_area,
                                            loss_smooth))
            
            writer.add_scalars('Train/loss', {'loss_dist': loss_dist*weight_distan,
                                              'loss_area': loss_area*weight_area,
                                              'loss_smooth': loss_smooth*weight_smooth},
                                              epoch*len(train_dataloader)+batch_idx)
            
            if loss_dist > 0.6:
                print(file)
            
            
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), model_file)
            
        
        torch.save(model.state_dict(), model_file)
        

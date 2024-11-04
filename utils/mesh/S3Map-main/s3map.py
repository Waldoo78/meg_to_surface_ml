#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 01:13:04 2021

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""
import argparse
import numpy as np
import glob
import os
import torch
import math

from s3pipe.models.models import SUnet
from s3pipe.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from s3pipe.utils.utils import get_neighs_order, get_sphere_template, get_neighs_faces
from s3pipe.surface.s3map import ResampledInnerSurf, ResampledInnerSurfVtk, computeMetrics_torch,\
    moveOrigSphe, computeAndWriteDistortionOnOrigSphe, \
        computeAndWriteDistortionOnRespSphe
from s3pipe.surface.prop import countNegArea
from s3pipe.utils.vtk import read_vtk, write_vtk

abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Perform spherical mapping of cortical surfaces with minimal metric distortion. '+\
                                     'It needs the initially spherical mapped and resampled surface using initial_spherical_mapping.py, ' +\
                                     'and its corresponding inner surface in .vtk format, the vtk files should contain vertices and faces fields.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resp_inner_surf', default=None, help="filename of the resampled inner surface, should end in *.inflated.SIP.RespInner.vtk. "+\
                        "Its correponding original inner surface filename should be resp_inner_surf.replace('.inflated.SIP.RespInner.vtk', '.vtk'). " +\
                        "Its correponding initial spherical surface filename should be resp_inner_surf.replace('.inflated.SIP.RespInner.vtk', '.inflated.SIP.vtk').")
    parser.add_argument('--files_pattern', default=None, help="pattern of resampled inner surface files, this can help process multiple files in one command. "+\
                                                              "Either single file or files pattern should be given. "+\
                                                              "Note the pattern needs to be quoted for python otherwise it will be parsed by the shell by default.")
    parser.add_argument('--hemi', help="hemisphere, lh or rh", choices=['lh', 'rh'], required=True)
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')
    parser.add_argument('--model_path', default=None, help="the folder containing all models, if not given, it will be ./pretrained_models ")
    parser.add_argument('--save_interim_results', default='True', help="save intermediate results or not, if Ture, there will be many results generated and need more storage", 
                        choices=['True', 'False'])
    args = parser.parse_args()
    resp_inner_surf = args.resp_inner_surf
    files_pattern = args.files_pattern
    hemi = args.hemi
    device = args.device
    model_path = args.model_path
    save_interim_results = args.save_interim_results == 'True'
    print('\n------------------------------------------------------------------')
    print('Distortion correction for the initial spherical mapped surface...')
    print('resp_inner_surf:', resp_inner_surf)
    print('files_pattern:', files_pattern)
    print('hemi:', hemi)
    print('save_interim_results:', save_interim_results)
    if model_path is None:
        model_path = abspath + '/pretrained_models'
    print('model_path:', model_path)
    
    # check device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    
    if resp_inner_surf is None and files_pattern is None:
        raise NotImplementedError('Either single resp_inner_surf file or files pattern should be given.')
    if resp_inner_surf is not None:
        files = [resp_inner_surf]
    else:
        files = sorted(glob.glob(files_pattern))
    print('files:', files)
    print('len(files):', len(files))
    
    dataset = 'HighRes'   # 'HighRes' for dataset BCP and HCP with vertices number between 140000 - 250000, 'LowRes' for dataset dHCP with vertices between 30000 to 140000
    weight_distan = 1.0
    weight_area = 0.1
    weight_smooth = 10.0
    batch_size = 1
    n_vertexs = [10242, 40962, 163842] #A mettre en argumant de arg.all dans le futur

    # start distortion correction for each level
    for n_vertex in n_vertexs:
        n_level = n_vertexs.index(n_vertex)
        print('\n------------------------------------------------------------------')
        print("Starting distortion correction on", n_level+1, "-th level with", n_vertex, "vertices.")
    
        if n_vertex == 10242:
            deform_scale = 10.0
            level = 6
            n_res = 4 
            resp_inner_file = files
        elif n_vertex == 40962:
            deform_scale = 10.0
            level = 7
            n_res = 4 
            resp_inner_file = [ x.replace('.SIP.RespInner.vtk', '.SIP.10242moved.RespInner.vtk') for x in files ]
        elif n_vertex == 163842:
            deform_scale = 20.0
            level = 8
            n_res = 5 
            resp_inner_file = [ x.replace('.SIP.RespInner.vtk', '.SIP.10242moved.40962moved.RespInner.vtk') for x in files ]
        else:
            raise NotImplementedError('Error')
        
                
        test_dataset = ResampledInnerSurfVtk(resp_inner_file, n_vertex)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        model = SUnet(in_ch=12, out_ch=2, level=level, n_res=n_res, rotated=0, complex_chs=32)
        model.to(device)
        model_file = os.path.join(model_path, 'S3Map_'+ dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                                                 '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl')
        print('Loading pre-trained model:', model_file)
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_file))
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        print('Loading pre-trained model done.')
        
        neigh_orders = get_neighs_order(n_vertex)
        neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_orders[:, 0:6]), axis=1)
        template = get_sphere_template(n_vertex)
        fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
        bi_inter = get_bi_inter(n_vertex, device)[0]
        En = getEn(n_vertex, device)[0]
        neighs_faces = get_neighs_faces(n_vertex)
        
        model.eval()
        for batch_idx, (inner_vert, file) in enumerate(test_dataloader):
            file = file[0]
            print("\nProcessing surface:", file)
            inner_vert = inner_vert.squeeze().to(device)
    
            with torch.no_grad():
                inner_dist, inner_area, _ = computeMetrics_torch(inner_vert, neigh_sorted_orders, device)
                feat = torch.cat((inner_dist, inner_area), 1)  # features
                feat = feat.permute(1,0).unsqueeze(0)
                deform_2d = model(feat) / deform_scale
                deform_2d = deform_2d[0,:].permute(1,0)
                deform_3d = convert2DTo3D(deform_2d, En, device)
       
                # diffeomorphic implementation
                velocity_3d = deform_3d/math.pow(2, 6)
                moved_sphere_loc = diffeomorp_torch(fixed_xyz, velocity_3d, 
                                                  num_composition=6, bi=True, 
                                                  bi_inter=bi_inter, 
                                                  device=device)
            print('Predicting spherical deformation field done.')
            
            resp_inner_surf = read_vtk(file)
            if save_interim_results:
                orig_sphere = {'vertices': fixed_xyz.cpu().numpy() * 100.0,
                               'faces': template['faces'],
                               'sulc': resp_inner_surf['sulc'][0:n_vertex],
                               'deformation': deform_3d.cpu().numpy() * 100.0}
                write_vtk(orig_sphere, file.replace('RespInner.vtk', 
                                                    'RespSphe.'+ str(n_vertex) +'deform.vtk'))
            
            moved_surf = {'vertices': moved_sphere_loc.cpu().numpy() * 100.0,
                          'faces': template['faces'],
                          'sulc': resp_inner_surf['sulc'][0:n_vertex],
                          }
            neg_area = countNegArea(moved_surf['vertices'], moved_surf['faces'][:, 1:])
            print("Negative areas of the deformation:", neg_area)
            if save_interim_results:
                write_vtk(moved_surf, file.replace('RespInner.vtk', 
                                               'RespSphe.'+ str(n_vertex) +'moved.vtk'))
        

            # postprocessing
            if n_vertex == 10242:
                orig_sphe_surf = read_vtk(file.replace('.SIP.RespInner.vtk', '.SIP.vtk'))
                orig_sphe_moved_name = file.replace('.SIP.RespInner.vtk', '.SIP.10242moved.vtk')
                orig_inner_surf = read_vtk(file.replace('.inflated.SIP.RespInner.vtk', '.vtk'))
            elif n_vertex == 40962:
                orig_sphe_surf = read_vtk(file.replace('.SIP.10242moved.RespInner.vtk', '.SIP.10242moved.vtk'))
                orig_sphe_moved_name = file.replace('.SIP.10242moved.RespInner.vtk', '.SIP.10242moved.40962moved.vtk')
                orig_inner_surf = read_vtk(file.replace('.inflated.SIP.10242moved.RespInner.vtk', '.vtk'))
            else:
                orig_sphe_surf = read_vtk(file.replace('.SIP.10242moved.40962moved.RespInner.vtk', '.SIP.10242moved.40962moved.vtk'))
                orig_sphe_moved_name = file.replace('.SIP.10242moved.40962moved.RespInner.vtk', '.SIP.10242moved.40962moved.163842moved.vtk')
                orig_inner_surf = read_vtk(file.replace('.inflated.SIP.10242moved.40962moved.RespInner.vtk', '.vtk'))
            
            print("\nMoving original sphere according to ico sphere deformation...")
            orig_sphere_moved = moveOrigSphe(template, orig_sphe_surf, moved_surf, orig_inner_surf, neighs_faces, orig_sphe_moved_name)
            print("Moving original sphere done!")
             
            if save_interim_results:
                computeAndWriteDistortionOnOrigSphe(orig_sphere_moved, orig_inner_surf, orig_sphe_moved_name)
            else:
                orig_sphere_surf = {'vertices': orig_sphere_moved,
                                    'faces': orig_inner_surf['faces'],
                                    'sulc':  orig_inner_surf['sulc'],
                                    'curv':  orig_inner_surf['curv']}
                write_vtk(orig_sphere_surf, orig_sphe_moved_name)
            print("computeAndWriteDistortionOnOrigSphe done!")
            
            if n_vertex!=40962:
                print("\nResampling inner surface...")
                template_40962 = get_sphere_template(40962)
                computeAndWriteDistortionOnRespSphe(orig_sphere_moved, template_40962, 
                                                    orig_inner_surf, orig_sphe_moved_name.replace('.vtk', '.RespInner.vtk'),
                                                    compute_distortion=save_interim_results)
                print("computeAndWriteDistortionOnRespSphe done!")

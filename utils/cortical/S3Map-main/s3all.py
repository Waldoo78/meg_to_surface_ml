#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:36:22 2021

@author: Fenqiang Zhao
contact: zhaofenqiang0221@gmail.com
"""

import glob
import argparse
import os

abspath = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Superfast spherical surface pipeline \n'+\
                                   '1. Computing mean curvature and area \n'+\
                                   '2. Inflating the inner surface and computing sulc \n'+\
                                   '3. Initial spherical mapping \n'+\
                                   '4. Distortion correction for initial spherical surface.',
                                   formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--input', '-i', required=True, 
                       help="the input inner surface (white matter surface) in vtk format. "+\
                            "Filename must contain 'lh' or 'rh' to identify hemisphere.")
    
    parser.add_argument('--device', default='CPU', choices=['GPU', 'CPU'],
                       help='Device to use for computation (default: CPU)')
    
    parser.add_argument('--save_interim_results', default='False', 
                       choices=['True', 'False'],
                       help="save intermediate results or not")

    args = parser.parse_args()
    input_name = args.input
        
    if 'lh' in input_name:
        hemi = 'lh'
    elif 'rh' in input_name:
        hemi = 'rh'
    else:
        raise ValueError('Input filename must contain lh or rh to identify hemisphere.')
    
    save_interim_results = args.save_interim_results == 'True'
    device = args.device
    
    print('\nProcessing input surface:', input_name)
    print('hemi:', hemi)
    print('device:', device)
    print('save_interim_results:', save_interim_results)
    
    # compute curv and area
    os.system(f"python {os.path.join(abspath, 'compute_curv_area.py')} -i {input_name}")
    
    # inflate and compute sulc
    os.system(f"python {os.path.join(abspath, 's3inflate.py')} -i {input_name}")
   
    # initial spherical mapping and resampling
    os.system(f"python {os.path.join(abspath, 'initial_spherical_mapping.py')} -i {input_name.replace('.vtk', '.inflated.vtk')}")
   
    # distortion correction
    os.system(f"python {os.path.join(abspath, 's3map.py')} " + \
             f"--resp_inner_surf {input_name.replace('.vtk', '.inflated.SIP.RespInner.vtk')} " + \
             f"--hemi {hemi} " + \
             f"--device {device} " + \
             f"--save_interim_results {args.save_interim_results}")
   
    if not save_interim_results:
        print('\nremoving intermediate results...')
        for f in glob.glob(input_name.replace('.vtk', '.*.RespSphe.vtk')):
            os.remove(f)
        for f in glob.glob(input_name.replace('.vtk', '.*.RespInner.vtk')):
            os.remove(f)
        print('Done.')
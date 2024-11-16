#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:36:22 2021

@author: Fenqiang Zhao
contact: zhaofenqiang0221@gmail.com
"""

import sys

import argparse
import time
import numpy as np

from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.surface.s3map import projectOntoSphere
from s3pipe.utils.interp_numpy import resampleSphereSurf
from s3pipe.utils.utils import get_sphere_template


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Initial spherical mapping and resampling. '+\
                                     'Output names will be automatically set to input.SIP.vtk and input.SIP.Resp.vtk',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i',  required=True, 
                        help="a sufficently inflated surface in vtk format, containing vertices and faces")
    parser.add_argument('--inner_surf', default=None, help="the inner surface filename. It will be resampled according to the initially mapped spherical surface.")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    input_name = args.input
    inner_name = args.inner_surf
   
    print('\n------------------------------------------------------------------') 
    print('Initial spherical mapping and resampling...')
    print('Input inflated surface:', input_name)
    if inner_name:
        inner_surf_vtk = read_vtk(inner_name)
        print('Input inner surface:', input_name)
    else:
        inner_surf_vtk = read_vtk(input_name.replace('.inflated.vtk', '.vtk'))
        print('Input inner surface:', input_name.replace('.inflated.vtk', '.vtk'))
    
    surf_vtk = read_vtk(input_name)
    print('Reading input surface...')

    sphere_0_ver = projectOntoSphere(surf_vtk['vertices'])
    initSphe_surf_vtk = {'vertices': sphere_0_ver,
                         'faces': inner_surf_vtk['faces'],
                         'sulc': inner_surf_vtk['sulc'],
                         'curv': inner_surf_vtk['curv'],
                         'area': inner_surf_vtk['area']}
    print('Initial spherical mapping done.')
    print('Saving initial spherical surface to', 
          input_name.replace('.vtk', '.SIP.vtk')) 
    write_vtk(initSphe_surf_vtk, input_name.replace('.vtk', '.SIP.vtk'))

    # resample the initial sphere surface    
    print('\n------------------------------------------------------------------')
    print('Resampling the initial spherical surface...')
    t1 = time.time()
    template_surf = get_sphere_template(40962)
    resampled_features = resampleSphereSurf(initSphe_surf_vtk['vertices'],
                                            template_surf['vertices'], 
                                            np.concatenate((initSphe_surf_vtk['sulc'][:, np.newaxis],
                                                            initSphe_surf_vtk['curv'][:, np.newaxis],
                                                            inner_surf_vtk['vertices']), axis=1),
                                            faces=surf_vtk['faces'][:, 1:])
    t2 = time.time()
    print('Resampling done, took {:.1f} s'.format(t2-t1))

    print("\nSaving resampled spherical surface to", 
          input_name.replace('.vtk', '.SIP.RespSphe.vtk'))
    resampled_spheSurf_vtk = {'vertices': template_surf['vertices'],
                              'faces': template_surf['faces'],
                              'sulc': resampled_features[:,0],
                              'curv': resampled_features[:,1]
                              }
    write_vtk(resampled_spheSurf_vtk, input_name.replace('.vtk', '.SIP.RespSphe.vtk'))
    print("Saving resampled inner surface to",
          input_name.replace('.vtk', '.SIP.RespInner.vtk'))
    resampled_innerSurf_vtk = {'vertices': resampled_features[:, -3:],
                                'faces': template_surf['faces'],
                                'sulc': resampled_features[:,0],
                                'curv': resampled_features[:,1]
                              }
    write_vtk(resampled_innerSurf_vtk, input_name.replace('.vtk', '.SIP.RespInner.vtk'))



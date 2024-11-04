#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:36:22 2021

@author: Fenqiang Zhao
contact: zhaofenqiang0221@gmail.com
"""

import sys

sys.path.append('/proj/ganglilab/users/Fenqiang/sunetpkg_py39/lib/python3.9/site-packages')
sys.path.append('/nas/longleaf/rhel8/apps/python/3.9.6/lib/python3.9/site-packages')


import argparse
import time
import os

from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.surface.inflate import InflateSurface
from s3pipe.surface.surf import Surface


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='inflate a cortical surface until either reaching the max iteration number ' + \
                                     'or the projected sphere has no self-intersection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', default=None, required=True, 
                        help="the input inner surface in vtk format, containing vertices and faces")
    parser.add_argument('--num_iter', default=600, help='The number of inflations.')
    parser.add_argument('--lambda_s','-lambda_s', default=1.0, help='weight for spring term, 1-lambda_s for preserving metrics (J_d term), ' +\
                        'default=1.0 since we will mimimize J_d on sphere later more conveniently and faster.')
    parser.add_argument('--output', '-o', default=None, help="the output inflated surface filename, if not given, it will be input.inflated.vtk")
    parser.add_argument('--save_sulc', default=True, help="save sulc or not, if not, inflation will be much faster")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    input_name = args.input
    output_name = args.output
    if output_name == None:
        output_name = os.path.splitext(input_name)[0] + '.inflated.vtk'
    num_iter = args.num_iter
    lambda_s = args.lambda_s
    save_sulc = args.save_sulc
        
    # inflate and compute sulc
    print('\n------------------------------------------------------------------')
    print('Inflating the inner surface and computing average convexity (sulc)')
    print('input:', input_name)
    print('output:', output_name)
    t1 = time.time()
    inner_surf_vtk = read_vtk(input_name)
    surf = Surface(inner_surf_vtk['vertices'], inner_surf_vtk['faces'])
    t2 = time.time()
    print('Reading surface and initializing surface graph done, took {:.1f} s'.format(t2-t1))

    print()  
    t1 = time.time()
    inflation_params = {'max_iter_num': int(num_iter),
                        'lambda': float(lambda_s),
                        'save_sulc': save_sulc,
                        'n_averages': 4,
                        'max_grad': 1.0,
                        'min_neg_area_num': 50,
                        'min_proj_iter': 200}
    InflateSurface(surf, inflation_params)
    print('Saving sulc to input inner surface', input_name)
    inner_surf_vtk['sulc'] = surf.sulc
    write_vtk(inner_surf_vtk, input_name)
    print('Saving inflated surface to', input_name.replace('.vtk', '.inflated.vtk'))
    inflated_surf_vtk = {'vertices': surf.vertices,
                         'faces': inner_surf_vtk['faces'],
                         'sulc': surf.sulc}
    
    write_vtk(inflated_surf_vtk, input_name.replace('.vtk', '.inflated.vtk'))
    t2 = time.time()
    print('Inflation done, took {:.1f} s'.format(t2-t1))
    
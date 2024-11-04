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
import s3pipe.surface.prop as sprop
from s3pipe.surface.surf import Surface


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Compute mean curvature and vertex-wise area',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', default=None, required=True, 
                        help="the input inner surface in vtk format, containing vertices and faces")
    parser.add_argument('--avg_iter', '-a', default=6,
                        help="the iteration number for averaging/smoothing the mean curvature feature")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    input_name = args.input
    avg_iter = int(args.avg_iter)
         
    print('\n------------------------------------------------------------------')
    print('Computing mean curvature and vertex-wise area...')
    print('Reading input surface:', input_name)
    t1 = time.time()
    inner_surf_vtk = read_vtk(input_name)
    surf = Surface(inner_surf_vtk['vertices'], inner_surf_vtk['faces'])
    t2 = time.time()
    print('Reading surface and initializing surface graph done, took {:.1f} s'.format(t2-t1))
    
    # compute curvature
    print('\nComputing mean curvature...')
    t1 = time.time()
    surf.curv = sprop.computeMeanCurvature(surf, dist=0.15)
    # surf.normalizeCurvature()
    surf.averageCurvature(avg_iter)
    print('Saving mean curvature to original input inner surface as a scalar property field.')
    inner_surf_vtk['curv'] = surf.curv
    # write_vtk(inner_surf_vtk, input_name)
    t2 = time.time()
    print('Computing mean curvature done, took {:.1f} s'.format(t2-t1))
    
    
    # compute area
    print('\nComputing vertex-wise area...')
    t1 = time.time()
    surf.orig_vertex_area = sprop.computeVertexArea(surf)
    print('Saving vertex-wise area to original input inner surface as a scalar property field.')
    inner_surf_vtk['area'] = surf.orig_vertex_area
    write_vtk(inner_surf_vtk, input_name)
    t2 = time.time()
    print('Computing vertex-wise area done, took {:.1f} s'.format(t2-t1))
    
    if np.mean(np.abs(surf.curv)) > 0.5:
        raise Warning('Very large and noisy curv computed. The surface may be reconstructed incorrectly.')

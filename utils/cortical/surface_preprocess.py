import os
import tempfile
import numpy as np
import pyvista as pv
from utils.file_manip import vtk_processing
from utils.file_manip.Matlab_to_array import load_faces, load_vertices


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_spherical_projection(mesh_data, hemisphere='lh'):
    """Projects a mesh onto a sphere at 40962 vertices using S3MAP.
    
    Args:
        mesh_data: Tuple of (coordinates, triangles)
        hemisphere: 'lh' or 'rh' for left or right hemisphere
    
    Returns:
        Tuple (sphere_coords, sphere_triangles) of the spherical projection
    """
    if hemisphere not in ['lh', 'rh']:
        raise ValueError("hemisphere must be 'lh' or 'rh'")
   
    coords, triangles = mesh_data
    print(f"Input mesh size: {len(coords)} vertices")
    
    # Setup temporary files
    temp_dir = tempfile.gettempdir()
    input_vtk = os.path.join(temp_dir, f"{hemisphere}.brain.vtk")
    sphere_vtk = input_vtk.replace('.vtk', '.inflated.SIP.10242moved.RespSphe.40962moved.vtk')
    
    # Save input mesh
    vtk_processing.save_to_vtk(coords, triangles, input_vtk)
    
    # Run S3MAP projection
    abs_path = os.path.abspath(input_vtk)
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "cortical", "S3MAP-main", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True --device CPU"
    print(f"Executing: {cmd}")
    os.system(cmd)

    # Load results
    if not os.path.exists(sphere_vtk):
        print(f"Looking for file: {sphere_vtk}")
        raise FileNotFoundError("S3MAP sphere output file not found")

    sphere_coords, sphere_triangles = vtk_processing.vtk_mesh_to_array(sphere_vtk)
    
    return sphere_coords, sphere_triangles

def get_resampled_inner_surface(mesh_data, hemisphere='lh'):
    """Gets the resampled inner surface at 40962 vertices using S3MAP.
    
    Args:
        mesh_data: Tuple of (coordinates, triangles)
        hemisphere: 'lh' or 'rh' for left or right hemisphere
    
    Returns:
        Tuple (target_coords, target_triangles) of the resampled inner surface
    """
    if hemisphere not in ['lh', 'rh']:
        raise ValueError("hemisphere must be 'lh' or 'rh'")
   
    coords, triangles = mesh_data
    print(f"Input mesh size: {len(coords)} vertices")
    
    # Setup temporary files
    temp_dir = tempfile.gettempdir()
    input_vtk = os.path.join(temp_dir, f"{hemisphere}.brain.vtk")
    target_vtk = input_vtk.replace('.vtk', '.inflated.SIP.10242moved.RespInner.vtk')
    
    # Save input mesh
    vtk_processing.save_to_vtk(coords, triangles, input_vtk)
    
    # Run S3MAP projection
    abs_path = os.path.abspath(input_vtk)
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "cortical", "S3MAP-main", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True --device CPU"
    print(f"Executing: {cmd}")
    os.system(cmd)

    # Load results
    if not os.path.exists(target_vtk):
        print(f"Looking for file: {target_vtk}")
        raise FileNotFoundError("S3MAP inner surface output file not found")

    target_coords, target_triangles = vtk_processing.vtk_mesh_to_array(target_vtk)
    
    return target_coords, target_triangles

def smooth_surface(vertices, faces, n_iterations=5, relaxation_factor=0.5):
    """
    Smooth a surface mesh using PyVista's smoothing function.
    """
    mesh = pv.PolyData(vertices, vtk_processing.convert_triangles_to_pyvista(faces))
    smoothed_mesh = mesh.smooth(n_iter=n_iterations, 
                              relaxation_factor=relaxation_factor,
                              feature_smoothing=False,
                              boundary_smoothing=True,
                              edge_angle=100,
                              feature_angle=100)
    return smoothed_mesh.points

def merge_hemis(hemi_lh, hemi_rh):
    """
    Merge left and right hemispheres while preserving their relative positions.
    
    Args:
        hemi_lh (tuple): (coordinates, triangles) for left hemisphere
        hemi_rh (tuple): (coordinates, triangles) for right hemisphere
        
    Returns:
        tuple: (merged coordinates, merged triangles)
    """
    lh_coords, lh_tris = hemi_lh
    rh_coords, rh_tris = hemi_rh
    
    merged_coords = np.vstack([lh_coords, rh_coords])
    rh_tris_adjusted = rh_tris + len(lh_coords)
    merged_tris = np.vstack([lh_tris, rh_tris_adjusted])
    
    return merged_coords, merged_tris

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
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "cortical", "S3Map", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True"
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
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "cortical", "S3Map", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True"
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



def process_cortical_surfaces(input_folder, output_folder, n_subjects=None):
    """
    Process cortical surfaces from input folder and save resampled versions in output folder
    using S3MAP for spherical projection.
    
    Parameters
    ----------
    input_folder : str
        Path to the main input folder containing subject subfolders with .mat files
        (lh_vertices.mat, lh_faces.mat, rh_vertices.mat, rh_faces.mat)
    output_folder : str
        Path to the output folder where processed data will be saved
        (lh_resampled.npz, rh_resampled.npz)
    n_subjects : int, optional
        Number of subjects to process. If None, process all subjects.
        If specified, will process the first n_subjects that haven't been processed yet.
    """
    
    def load_and_process_hemisphere(vertices_file, faces_file, hemisphere):
        """Helper function to process one hemisphere"""
        faces = load_faces(faces_file)
        vertices = load_vertices(vertices_file)
        coords, tris = get_resampled_inner_surface((vertices, faces), hemisphere)
        center = np.mean(coords, axis=0)
        coords = coords - center
        return coords, tris, center

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Counter for processed subjects
    processed_count = 0

    # Process each subject folder
    for subject in os.listdir(input_folder):
        # Stop if we've processed the requested number of subjects
        if n_subjects is not None and processed_count >= n_subjects:
            print(f"\nReached requested number of subjects ({n_subjects}). Stopping.")
            break

        input_subject_path = os.path.join(input_folder, subject)
        
        # Skip if not a directory
        if not os.path.isdir(input_subject_path):
            continue
            
        # Check if this subject has already been processed
        output_subject_path = os.path.join(output_folder, subject)
        lh_output = os.path.join(output_subject_path, "lh_resampled.npz")
        rh_output = os.path.join(output_subject_path, "rh_resampled.npz")
        
        # Skip if both hemispheres are already processed
        if os.path.exists(lh_output) and os.path.exists(rh_output):
            print(f"Subject {subject} already processed. Skipping.")
            continue

        print(f"Processing subject: {subject} ({processed_count + 1}" + 
              f"{'/' + str(n_subjects) if n_subjects else ''})")
        
        # Create subject output directory
        os.makedirs(output_subject_path, exist_ok=True)
        
        success = True  # Track if both hemispheres are processed successfully

        # Process left hemisphere if needed
        if not os.path.exists(lh_output):
            try:
                left_vertices_file = os.path.join(input_subject_path, "lh_vertices.mat")
                left_faces_file = os.path.join(input_subject_path, "lh_faces.mat")
                
                coords, tris, center = load_and_process_hemisphere(
                    left_vertices_file, 
                    left_faces_file, 
                    'lh'
                )
                
                np.savez(lh_output, coords=coords, tris=tris, center=center)
                print(f"  Left hemisphere processed successfully")
                
            except Exception as e:
                print(f"  Error processing left hemisphere: {str(e)}")
                success = False
        
        # Process right hemisphere if needed
        if not os.path.exists(rh_output):
            try:
                right_vertices_file = os.path.join(input_subject_path, "rh_vertices.mat")
                right_faces_file = os.path.join(input_subject_path, "rh_faces.mat")
                
                coords, tris, center = load_and_process_hemisphere(
                    right_vertices_file, 
                    right_faces_file, 
                    'rh'
                )
                
                np.savez(rh_output, coords=coords, tris=tris, center=center)
                print(f"  Right hemisphere processed successfully")
                
            except Exception as e:
                print(f"  Error processing right hemisphere: {str(e)}")
                success = False

        # Only increment counter if at least one hemisphere was processed successfully
        if success:
            processed_count += 1

    print(f"\nProcessing complete. {processed_count} subjects processed.")


if __name__=="__main__":
    from utils.file_manip.vtk_processing import convert_triangles_to_pyvista
    import pyvista as pv 
    folder_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC110033"
    left_vertices_file = os.path.join(folder_path, "lh_vertices")
    left_faces_file = os.path.join(folder_path, "lh_faces")
    output_file = os.path.join(folder_path, "lh_resampled.npz")
    left_faces = load_faces(left_faces_file)
    left_vertices = load_vertices(left_vertices_file)
    coords, tris=get_resampled_inner_surface((left_vertices, left_faces), hemisphere='lh')
    mesh=pv.PolyData(coords, convert_triangles_to_pyvista(tris))
    p=pv.Plotter()
    p.add_mesh(mesh, show_edges=True)
    p.show()
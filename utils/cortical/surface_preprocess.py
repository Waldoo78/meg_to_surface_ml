import os
import tempfile
import numpy as np
import pyvista as pv
from utils.file_manip import vtk_processing
from utils.file_manip.Matlab_to_array import load_faces, load_vertices

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# S3MAP preprocess 
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


import os
import numpy as np
import pyvista as pv
from utils.file_manip import vtk_processing
from utils.file_manip.Matlab_to_array import load_faces, load_vertices

def check_negative_triangles(coords, triangles):
    """Checks the number of triangles with negative areas"""
    mesh = pv.PolyData(coords, vtk_processing.convert_triangles_to_pyvista(triangles))
    mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)
    # Compute triangle areas
    areas = mesh.compute_cell_sizes().cell_data['Area']
    negative_count = np.sum(areas < 0)
    total_count = len(areas)
    print(f"Triangles with negative areas: {negative_count}/{total_count} ({negative_count/total_count*100:.2f}%)")
    return negative_count, total_count

def compare_surfaces(original_vertices_file, original_faces_file, resampled_npz_file):
    """
    Compare the original surface with its resampled version.
    
    Args:
        original_vertices_file: Path to the original vertices file
        original_faces_file: Path to the original faces file
        resampled_npz_file: Path to the resampled surface npz file
    """
    # Load the original surface
    original_vertices = load_vertices(original_vertices_file)
    original_faces = load_faces(original_faces_file)
    
    # Load the resampled surface 
    resampled_data = np.load(resampled_npz_file)
    resampled_vertices = resampled_data['coords'] + resampled_data['center']
    resampled_faces = resampled_data['tris']
    
    # Basic stats
    print("=== Surface Statistics ===")
    print(f"Original surface: {len(original_vertices)} vertices, {len(original_faces)} triangles")
    print(f"Resampled surface: {len(resampled_vertices)} vertices, {len(resampled_faces)} triangles")
    
    # Check triangles with negative areas 
    print("\n=== Checking triangles with negative areas ===")
    print("Original surface:")
    check_negative_triangles(original_vertices, original_faces)
    print("Resampled surface:")
    check_negative_triangles(resampled_vertices, resampled_faces)
    
    # Compute normals and check their orientation
    print("\n=== Checking normal orientation ===")
    def check_normals_orientation(vertices, faces, label):
        mesh = pv.PolyData(vertices, vtk_processing.convert_triangles_to_pyvista(faces))
        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        normals = mesh.cell_data['Normals']
        
        # Estimate approximate center (can be improved for a brain)
        center = np.mean(vertices, axis=0)
        
        # For each triangle, check if its normal points outward
        cell_centers = mesh.cell_centers().points
        vectors_to_center = cell_centers - center  # From brain center to triangle center
        
        # Calculate dot products
        # A positive dot product indicates vectors pointing in roughly the same direction
        dot_products = np.sum(vectors_to_center * normals, axis=1)
        
        outward_count = np.sum(dot_products > 0)
        inward_count = np.sum(dot_products < 0)
        total_count = len(dot_products)
        
        print(f"{label}:")
        print(f"  Outward normals: {outward_count}/{total_count} ({outward_count/total_count*100:.2f}%)")
        print(f"  Inward normals: {inward_count}/{total_count} ({inward_count/total_count*100:.2f}%)")
        
        return outward_count, inward_count, total_count
    
    check_normals_orientation(original_vertices, original_faces, "Original surface")
    check_normals_orientation(resampled_vertices, resampled_faces, "Resampled surface")
    
    # Visualization
    print("\n=== Visualization setup ===")
    print("You can visually compare both surfaces in an interactive window.")
    print("Original surface in white, resampled surface in red.")
    
    orig_mesh = pv.PolyData(original_vertices, vtk_processing.convert_triangles_to_pyvista(original_faces))
    resampled_mesh = pv.PolyData(resampled_vertices, vtk_processing.convert_triangles_to_pyvista(resampled_faces))
    
    p = pv.Plotter()
    p.add_mesh(orig_mesh, color='white', label='Original', opacity=0.7)
    p.add_mesh(resampled_mesh, color='red', label='Resampled', opacity=0.7)
    p.add_legend()
    p.show()
    
# Usage example
if __name__ == "__main__":
    from utils.file_manip.vtk_processing import convert_triangles_to_pyvista
    import pyvista as pv
    
    folder_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC720071"
    left_vertices_file = os.path.join(folder_path, "lh_vertices")
    left_faces_file = os.path.join(folder_path, "lh_faces")
    output_file = os.path.join(folder_path, "lh_resampled.npz")
    
    # Load original data
    left_faces = load_faces(left_faces_file)
    left_vertices = load_vertices(left_vertices_file)
    
    # Get resampled surface
    coords, tris = get_resampled_inner_surface((left_vertices, left_faces), hemisphere='lh')
    
    # Check triangles with negative areas
    print("\n=== Checking triangles with negative areas on resampled surface ===")
    check_negative_triangles(coords, tris)
    
    # Save results in NPZ format
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    np.savez(output_file, coords=coords_centered, tris=tris, center=center)
    print(f"Resampled surface saved to {output_file}")
    
    # Visualize result
    mesh = pv.PolyData(coords, convert_triangles_to_pyvista(tris))
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True)
    p.show()
    
    # If NPZ file already exists, do a complete comparison
    if os.path.exists(output_file):
        print("\n=== Comparison with original surface ===")
        compare_surfaces(left_vertices_file, left_faces_file, output_file)
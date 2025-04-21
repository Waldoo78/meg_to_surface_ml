import numpy as np
import os
import pickle
from src.cortical_transformation.reconstruction import reconstruct_brain
from utils.file_manip.vtk_processing import convert_triangles_to_pyvista

def reduce_mesh_using_matlab(vertices, faces, target_vertices):
    try:
        import matlab.engine
        
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        
        # Convert numpy arrays to MATLAB matrices
        matlab_vertices = matlab.double(vertices.tolist())
        # +1 for Python (0-based) to MATLAB (1-based) indexing
        matlab_faces = matlab.double((faces + 1).tolist())
        
        ratio = float(target_vertices) / vertices.shape[0]
        
        # Call reducepatch
        faces_reduced, vertices_reduced = eng.reducepatch(matlab_faces, matlab_vertices, ratio, nargout=2)
        
        # Convert back to numpy arrays and -1 for Python indexing
        vertices_reduced = np.array(vertices_reduced)
        faces_reduced = np.array(faces_reduced, dtype=np.int32) - 1
        
        eng.quit()
        
        return vertices_reduced, faces_reduced
        
    except Exception as e:
        if 'eng' in locals():
            eng.quit()
        raise RuntimeError(f"Error in reducepatch: {str(e)}")

def generate_and_save_surfaces(Y_lh_full, Y_rh_full, lmax, data_path, merge=True,
                               subject_folder=None, coeffs_lh=None, coeffs_rh=None, 
                               lh_center=None, rh_center=None):
    """
    Generate and save brain surfaces using spherical harmonics
    
    Parameters:
    -----------
    Y_lh_full, Y_rh_full : numpy arrays
        Spherical harmonics for left and right hemispheres
    lmax : int
        Maximum degree of spherical harmonics to use
    data_path : str
        Path to data directory containing triangle information
    merge : bool
        Whether to merge left and right hemispheres
    subject_folder : str or None
        Path to subject folder, needed if coeffs or centers not provided
    coeffs_lh, coeffs_rh : dict or None
        Coefficients for left and right hemispheres, loaded from subject_folder if None
    lh_center, rh_center : numpy arrays or None
        Centers for left and right hemispheres, loaded from subject_folder if None
    """
    # Load data if not provided
    if coeffs_lh is None or coeffs_rh is None:
        if subject_folder is None:
            raise ValueError("Either coeffs_lh/coeffs_rh or subject_folder must be provided")
        
        # Load subject coefficients
        with open(os.path.join(subject_folder, "coeffs_lh.pkl"), 'rb') as f:
            coeffs_lh_data = pickle.load(f)
        
        with open(os.path.join(subject_folder, "coeffs_rh.pkl"), 'rb') as f:
            coeffs_rh_data = pickle.load(f)
        
        # Use the full coefficients dictionaries
        coeffs_lh = coeffs_lh_data
        coeffs_rh = coeffs_rh_data
    
    if lh_center is None or rh_center is None:
        if subject_folder is None:
            raise ValueError("Either lh_center/rh_center or subject_folder must be provided")
        
        # Load centers
        lh_center = np.load(os.path.join(subject_folder, "lh_center.npz"))['center']
        rh_center = np.load(os.path.join(subject_folder, "rh_center.npz"))['center']
    
    # Load triangles for reconstruction
    fsav_data_lh = np.load(os.path.join(data_path, "lh_resampled.npz"))
    tris = fsav_data_lh['tris']

    # Get reconstructed surfaces
    if merge:
        reconstructed_coords, reconstructed_tris = reconstruct_brain(
            lh_center=lh_center,
            rh_center=rh_center,
            coeffs_lh=coeffs_lh,
            coeffs_rh=coeffs_rh,
            Y_lh_full=Y_lh_full,
            Y_rh_full=Y_rh_full,
            tris=tris,
            l=lmax,
            merge=True
        )
        # Decimate merged brain
        vertices_reduced, faces_reduced = reduce_mesh_using_matlab(
            reconstructed_coords, 
            reconstructed_tris,
            target_vertices=15000
        )
        return vertices_reduced, faces_reduced
    else:
        # Get separate hemispheres
        lh_coords, rh_coords, hemi_tris = reconstruct_brain(
            lh_center=lh_center,
            rh_center=rh_center,
            coeffs_lh=coeffs_lh,
            coeffs_rh=coeffs_rh,
            Y_lh_full=Y_lh_full,
            Y_rh_full=Y_rh_full,
            tris=tris,
            l=lmax,
            merge=False
        )
        
        # Calculate proportion for decimation
        total_vertices = len(lh_coords) + len(rh_coords)
        lh_target = int(15000 * (len(lh_coords) / total_vertices))
        rh_target = 15000 - lh_target  # ensures total is exactly 15000
        
        # Decimate each hemisphere
        lh_verts_reduced, lh_faces_reduced = reduce_mesh_using_matlab(
            lh_coords, hemi_tris, target_vertices=lh_target
        )
        rh_verts_reduced, rh_faces_reduced = reduce_mesh_using_matlab(
            rh_coords, hemi_tris, target_vertices=rh_target
        )
        
        return (lh_verts_reduced, lh_faces_reduced), (rh_verts_reduced, rh_faces_reduced)

if __name__=="__main__":
    import pyvista as pv
    import copy
    
    # Define paths
    PATHS = {
        'main': r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN",
        'fsaverage': r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\fsaverage",
        'harmonics': r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation\data"
    }
    
    sub_name = "sub-CC110033"
    PATHS['subject'] = os.path.join(PATHS['main'], sub_name)

    # Load hemisphere-specific harmonics
    Y_lh_full = np.load(os.path.join(PATHS['harmonics'], "Y_lh.npz"))['Y']
    Y_rh_full = np.load(os.path.join(PATHS['harmonics'], "Y_rh.npz"))['Y']

    # Example 1: Using subject_folder to load coefficients internally
    merged_result = generate_and_save_surfaces(
        Y_lh_full=Y_lh_full,
        Y_rh_full=Y_rh_full,
        lmax=30,
        data_path=PATHS['fsaverage'],
        subject_folder=PATHS['subject'],
        merge=True
    )
    
    # Example 2: Pre-loading coefficients and passing them explicitly
    # Load coefficients
    with open(os.path.join(PATHS['subject'], "coeffs_lh.pkl"), 'rb') as f:
        coeffs_lh = pickle.load(f)
    with open(os.path.join(PATHS['subject'], "coeffs_rh.pkl"), 'rb') as f:
        coeffs_rh = pickle.load(f)
    
    # Load centers
    lh_center = np.load(os.path.join(PATHS['subject'], "lh_center.npz"))['center']
    rh_center = np.load(os.path.join(PATHS['subject'], "rh_center.npz"))['center']
    
    # Optional: apply epsilon modification to coefficients
    epsilon = 0.01
    coeffs_lh_eps = copy.deepcopy(coeffs_lh)
    coeffs_rh_eps = copy.deepcopy(coeffs_rh)
    for l in [16, 17]:
        for m in range(2*l+1):
            coeffs_lh_eps["organized_coeffs"][l][m] += np.array([epsilon+1j*epsilon, epsilon+1j*epsilon, epsilon+1j*epsilon])
            coeffs_rh_eps["organized_coeffs"][l][m] += np.array([epsilon+1j*epsilon, epsilon+1j*epsilon, epsilon+1j*epsilon])
    
    # Get separate hemispheres with pre-loaded data
    hemispheres = generate_and_save_surfaces(
        Y_lh_full=Y_lh_full,
        Y_rh_full=Y_rh_full,
        lmax=40,
        data_path=PATHS['fsaverage'],
        coeffs_lh=coeffs_lh_eps,
        coeffs_rh=coeffs_rh_eps,
        lh_center=lh_center,
        rh_center=rh_center,
        merge=False
    )
    
    # Visualization of the merged result
    reconstructed_coords, reconstructed_tris = merged_result
    mesh = pv.PolyData(reconstructed_coords, convert_triangles_to_pyvista(reconstructed_tris))
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, color='white')
    p.show()
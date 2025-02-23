import numpy as np
import os
import pickle
from utils.cortical import spherical_harmonics as SH
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

def generate_and_save_surfaces(main_folder, Y_lh_full, Y_rh_full, coeffs_diff_lh, 
                            coeffs_diff_rh, lh_center, rh_center, lmax, sigma, 
                            fsaverage_path, merge):
    # Load fsaverage template data
    with open(os.path.join(fsaverage_path, "coeffs_lh.pkl"), 'rb') as f:
        coeffs_fsav_lh_full = pickle.load(f)['organized_coeffs']
    fsav_data_lh = np.load(os.path.join(fsaverage_path, "lh_resampled.npz"))
    tris = fsav_data_lh['tris']
    
    with open(os.path.join(fsaverage_path, "coeffs_rh.pkl"), 'rb') as f:
        coeffs_fsav_rh_full = pickle.load(f)['organized_coeffs']

    # Truncate fsaverage coefficients to lmax
    coeffs_fsav_lh = {l: coeffs_fsav_lh_full[l] for l in range(lmax + 1)}
    coeffs_fsav_rh = {l: coeffs_fsav_rh_full[l] for l in range(lmax + 1)}
                
    # Sum coefficients 
    coeffs_sum_lh = {
        0: coeffs_fsav_lh[0], 
        **{l: coeffs_fsav_lh[l] + coeffs_diff_lh['organized_coeffs'][l] 
           for l in range(1, lmax + 1)}
    }
    
    coeffs_sum_rh = {
        0: coeffs_fsav_rh[0],
        **{l: coeffs_fsav_rh[l] + coeffs_diff_rh['organized_coeffs'][l] 
           for l in range(1, lmax + 1)}
    }

    coeffs_lh = {'organized_coeffs': coeffs_sum_lh}
    coeffs_rh = {'organized_coeffs': coeffs_sum_rh}

    # Get reconstructed surfaces
    if merge==True:
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
    import time 
    import pyvista as pv
    main_folder = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN"
    data_path=r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation\data"
    fsaverage_path=r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\fsaverage"

    # Load hemisphere-specific harmonics and slice according to lmax (limited to lmax<=80)
    Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
    Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']

    sub_name="sub-CC110033"
    subject_folder=os.path.join(main_folder, sub_name)

    # Load coefficient differences
    with open(os.path.join(subject_folder, "coeffs_diff_lh.pkl"), 'rb') as f:
        coeffs_diff_lh = pickle.load(f)
    with open(os.path.join(subject_folder, "coeffs_diff_rh.pkl"), 'rb') as f:
        coeffs_diff_rh = pickle.load(f)

    # Load centers
    lh_center = np.load(os.path.join(subject_folder, "lh_center.npz"))['center']
    rh_center = np.load(os.path.join(subject_folder, "rh_center.npz"))['center']

    start_time = time.time()
    reconstructed_coords, reconstructed_tris = generate_and_save_surfaces(
        main_folder=main_folder,
        Y_lh_full=Y_lh_full,
        Y_rh_full=Y_rh_full,
        coeffs_diff_lh=coeffs_diff_lh,
        coeffs_diff_rh=coeffs_diff_rh,
        lh_center=lh_center,
        rh_center=rh_center,
        lmax=30,
        sigma=1e-4,
        fsaverage_path=fsaverage_path
    )
    execution_time = time.time() - start_time
    print(f"Temps d'exécution de generate_and_save_surfaces : {execution_time:.2f} secondes")

    # Visualisation
    mesh = pv.PolyData(reconstructed_coords, convert_triangles_to_pyvista(reconstructed_tris))
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, color='white')
    p.show()
import numpy as np
import pyvista as pv
import os
import scipy.io as sio
import pickle
from pathlib import Path

from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.mathutils import compute_vertex_normals, build_template_adjacency_two_hemis, compute_mean_curvature
from utils.cortical import spherical_harmonics as SH 
from utils.cortical import surface_preprocess as sp
from utils.file_manip.vtk_processing import convert_triangles_to_pyvista

def process_subjects(base_dir, data_path, subject_id=None, output_dir=None, sigma=1e-4, 
                   lmax=30, lambda_reg=0, max_order_coeff=80, visualize_first=False, save_surface=False):
    """
    Process all subjects or a specific subject with a logical sequence.
    
    Args:
        base_dir (str): Directory containing subject folders
        data_path (str): Path to template and harmonics data
        subject_id (str, optional): Process only this specific subject if provided
        output_dir (str, optional): Output directory. By default, saves in each subject folder.
        sigma (float): Smoothing parameter
        lmax (int): Maximum degree of spherical harmonics used for reconstruction
        lambda_reg (float): Regularization parameter
        max_order_coeff (int): Maximum order for coefficient storage
        visualize_first (bool): Visualize results for the first processed subject
        save_surface (bool): Save reconstructed surface in MATLAB format
    
    Returns:
        dict: Dictionary with processing results if a specific subject_id is provided
    """
    # Load templates and harmonics (common to all subjects)
    print("Loading templates and harmonics...")
    template_projection_lh = np.load(os.path.join(data_path, "lh_sphere_projection.npz"))
    template_projection_rh = np.load(os.path.join(data_path, "rh_sphere_projection.npz"))
    
    Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
    Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']
    
    # Cut harmonics for coefficient calculation up to max_order_coeff
    Y_lh_coeff = Y_lh_full[:, :(max_order_coeff+1)**2]
    Y_rh_coeff = Y_rh_full[:, :(max_order_coeff+1)**2]
    
    # Cut harmonics according to lmax for reconstruction
    Y_lh_recon = Y_lh_full[:, :(lmax+1)**2]
    Y_rh_recon = Y_rh_full[:, :(lmax+1)**2]
    
    # Find all subject directories or a specific subject
    subject_dirs = []
    if subject_id:
        folder_path = os.path.join(base_dir, subject_id)
        if os.path.isdir(folder_path):
            subject_dirs.append(folder_path)
        else:
            print(f"Subject directory {subject_id} not found!")
            return None
    else:
        for item in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, item)
            if os.path.isdir(folder_path) and item.startswith("sub-"):
                subject_dirs.append(folder_path)
    
    if not subject_dirs:
        print("No subject directories found!")
        return None
    
    print(f"Processing {len(subject_dirs)} subjects...")
    
    results = {}
    
    # Process each subject
    for i, subject_folder in enumerate(subject_dirs):
        subject_id = os.path.basename(subject_folder)
        print(f"\nProcessing subject {i+1}/{len(subject_dirs)}: {subject_id}")
        
        subject_results = {}
        
        try:
            # Check that necessary files exist
            required_files = [
                "lh_resampled.npz", "rh_resampled.npz",
                "lh_vertices.mat", "lh_faces.mat",
                "rh_vertices.mat", "rh_faces.mat"
            ]
            
            if not all(os.path.exists(os.path.join(subject_folder, f)) for f in required_files):
                print(f"Missing files for {subject_id}, skipping to next")
                continue
            
            # Load data for both hemispheres
            lh_resampled = np.load(os.path.join(subject_folder, "lh_resampled.npz"))
            rh_resampled = np.load(os.path.join(subject_folder, "rh_resampled.npz"))
            
            # Load original data for reference
            lh_vertices = load_vertices(os.path.join(subject_folder, "lh_vertices.mat"))
            lh_faces = load_faces(os.path.join(subject_folder, "lh_faces.mat"))
            rh_vertices = load_vertices(os.path.join(subject_folder, "rh_vertices.mat"))
            rh_faces = load_faces(os.path.join(subject_folder, "rh_faces.mat"))
            
            # Prepare tuples of resampled surfaces
            lh_surface_tuple = (lh_resampled['coords'], lh_resampled['tris'])
            rh_surface_tuple = (rh_resampled['coords'], rh_resampled['tris'])
            
            # 1. Calculate spherical harmonic coefficients
            print(f"Computing SH coefficients up to order {max_order_coeff}...")
            lh_coeffs_full = SH.compute_coefficients_SVD(Y_lh_coeff, lh_surface_tuple, max_order_coeff, lambda_reg)
            rh_coeffs_full = SH.compute_coefficients_SVD(Y_rh_coeff, rh_surface_tuple, max_order_coeff, lambda_reg)
            
            # Prepare dictionary for coefficients (exact structure)
            sh_coefficients = {
                'lh_coefficients': lh_coeffs_full,
                'rh_coefficients': rh_coeffs_full,
                'max_order': max_order_coeff,
                'lambda_reg': lambda_reg
            }
            
            # 2. Reconstruct surfaces using calculated coefficients (order lmax)
            print(f"Reconstructing surfaces with order {lmax}...")
            lh_reconstruction = SH.generate_surface(
                Y_lh_recon, 
                lmax, 
                sigma, 
                orders=lh_coeffs_full['organized_coeffs']
            )
            
            rh_reconstruction = SH.generate_surface(
                Y_rh_recon, 
                lmax, 
                sigma, 
                orders=rh_coeffs_full['organized_coeffs']
            )
            
            # Create reconstruction results for both hemispheres
            lh_results = {
                'resampled_surface': lh_surface_tuple,
                'reconstruction': lh_reconstruction,
                'coefficients': lh_coeffs_full,
                'original': {
                    'vertices': lh_vertices,
                    'faces': lh_faces
                }
            }
            
            rh_results = {
                'resampled_surface': rh_surface_tuple,
                'reconstruction': rh_reconstruction,
                'coefficients': rh_coeffs_full,
                'original': {
                    'vertices': rh_vertices,
                    'faces': rh_faces
                }
            }
            
            # 3. Merge hemispheres
            print("Merging hemispheres...")
            lh_reconstruction_centered = lh_reconstruction + lh_resampled['center']
            rh_reconstruction_centered = rh_reconstruction + rh_resampled['center']
            
            reconstructed_merged_coords, reconstructed_merged_tris = sp.merge_hemis(
                (lh_reconstruction_centered, lh_results['resampled_surface'][1]),
                (rh_reconstruction_centered, rh_results['resampled_surface'][1])
            )
            
            merged_results = {
                'reconstructed': {
                    'coords': reconstructed_merged_coords,
                    'tris': reconstructed_merged_tris
                },
                'lh_results': lh_results,
                'rh_results': rh_results
            }
            
            # 4. Analyze reconstruction results
            print("Analyzing results...")
            
            # Create faces for the full brain
            n_vertices_lh = len(template_projection_lh['sphere_coords'])
            rh_tris_adjusted = template_projection_rh['sphere_tris'] + n_vertices_lh
            full_faces = np.vstack([
                template_projection_lh['sphere_tris'],
                rh_tris_adjusted
            ])
            
            # Get resampled and reconstructed vertices
            vertices_resampled = np.vstack([
                lh_resampled['coords'] + lh_resampled['center'],
                rh_resampled['coords'] + rh_resampled['center']
            ])
            vertices_reconstructed = reconstructed_merged_coords
            
            # Build adjacency for calculations
            vertex_to_faces = build_template_adjacency_two_hemis(
                template_projection_lh['sphere_tris'],
                template_projection_rh['sphere_tris']
            )
            
            # Smooth the resampled surface
            vertices_resampled_smooth = sp.smooth_surface(
                vertices_resampled, 
                full_faces, 
                n_iterations=5, 
                relaxation_factor=0.5
            )
            
            # Calculate point-to-point distances
            distances = np.sqrt(np.sum((vertices_resampled_smooth - vertices_reconstructed)**2, axis=1)) * 1000  # Convert to mm
            
            # Calculate normals and analyze differences
            normals_resampled = compute_vertex_normals(vertices_resampled_smooth, full_faces, vertex_to_faces, n_rings=2)
            normals_reconstructed = compute_vertex_normals(vertices_reconstructed, full_faces, vertex_to_faces, n_rings=2)
            angles = np.degrees(np.arccos(np.clip(np.sum(normals_resampled * normals_reconstructed, axis=1), -1.0, 1.0)))
            
            # Curvature analysis
            curvature_resampled = compute_mean_curvature(vertices_resampled_smooth, full_faces, vertex_to_faces)
            curvature_reconstructed = compute_mean_curvature(vertices_reconstructed, full_faces, vertex_to_faces)
            curvature_diff = np.abs(curvature_resampled - curvature_reconstructed)
            
            # Prepare statistics
            stats = {
                'distances': {
                    'mean': np.mean(distances),
                    'max': np.max(distances),
                    'p95': np.percentile(distances, 95)
                },
                'angles': {
                    'mean': np.mean(angles),
                    'max': np.max(angles),
                    'p95': np.percentile(angles, 95),
                    'pct_over_45': 100*np.sum(angles > 45)/len(angles)
                },
                'curvature': {
                    'resampled_mean': np.mean(curvature_resampled),
                    'resampled_std': np.std(curvature_resampled),
                    'resampled_range': [np.min(curvature_resampled), np.max(curvature_resampled)],
                    
                    'reconstructed_mean': np.mean(curvature_reconstructed),
                    'reconstructed_std': np.std(curvature_reconstructed),
                    'reconstructed_range': [np.min(curvature_reconstructed), np.max(curvature_reconstructed)],
                    
                    'diff_mean': np.mean(curvature_diff),
                    'diff_std': np.std(curvature_diff),
                    'diff_max': np.max(curvature_diff),
                    'diff_p75': np.percentile(curvature_diff, 75),
                    'diff_p95': np.percentile(curvature_diff, 95)
                }
            }
            
            # Visualization only for the first subject or if requested
            if i == 0 and visualize_first:
                print("Visualizing results...")
                # Point-to-point distances
                pl = pv.Plotter(shape=(1, 2))
                mesh_resampled = pv.PolyData(vertices_resampled_smooth, convert_triangles_to_pyvista(full_faces))
                mesh_reconstructed = pv.PolyData(vertices_reconstructed, convert_triangles_to_pyvista(full_faces))
                
                pl.subplot(0, 0)
                pl.add_mesh(mesh_resampled, color='lightgray', show_edges=True, edge_color='black', line_width=1)
                pl.add_text("Resampled Surface", position='upper_edge')
                pl.view_isometric()
                
                pl.subplot(0, 1)
                pl.add_mesh(mesh_reconstructed, scalars=distances, cmap='viridis',
                           show_edges=True, edge_color='black', line_width=1,
                           scalar_bar_args={'title': 'Distance Error (mm)',
                                          'n_labels': 5})
                pl.add_text(f"Point-to-point distance errors\nMean: {np.mean(distances):.2f} mm\nMax: {np.max(distances):.2f} mm", 
                           position='upper_edge')
                pl.view_isometric()
                pl.link_views()
                pl.show()
            
            # Display statistics only for the first subject
            if i == 0:
                print("\n=== Analysis Summary ===")
                print("\nPoint-to-point distances (mm):")
                print(f"Mean: {stats['distances']['mean']:.2f}")
                print(f"Maximum: {stats['distances']['max']:.2f}")
                print(f"95th percentile: {stats['distances']['p95']:.2f}")

                print("\nNormal angles (degrees):")
                print(f"Mean: {stats['angles']['mean']:.2f}")
                print(f"Maximum: {stats['angles']['max']:.2f}")
                print(f"95th percentile: {stats['angles']['p95']:.2f}")
                print(f"Percentage >45°: {stats['angles']['pct_over_45']:.1f}%")

                print("\n=== Mean Curvature Analysis ===")
                print("\nResampled surface:")
                print(f"- Mean curvature: {stats['curvature']['resampled_mean']:.4f}")
                print(f"- Standard deviation: {stats['curvature']['resampled_std']:.4f}")
                print(f"- Min/Max: {stats['curvature']['resampled_range'][0]:.4f} / {stats['curvature']['resampled_range'][1]:.4f}")

                print("\nReconstructed surface:")
                print(f"- Mean curvature: {stats['curvature']['reconstructed_mean']:.4f}")
                print(f"- Standard deviation: {stats['curvature']['reconstructed_std']:.4f}")
                print(f"- Min/Max: {stats['curvature']['reconstructed_range'][0]:.4f} / {stats['curvature']['reconstructed_range'][1]:.4f}")

                print("\nCurvature differences:")
                print(f"- Mean absolute difference: {stats['curvature']['diff_mean']:.4f}")
                print(f"- Standard deviation of differences: {stats['curvature']['diff_std']:.4f}")
                print(f"- Maximum difference: {stats['curvature']['diff_max']:.4f}")
                print(f"- 75th percentile: {stats['curvature']['diff_p75']:.4f}")
                print(f"- 95th percentile: {stats['curvature']['diff_p95']:.4f}")
            
            # 5. Save results
            save_dir = output_dir if output_dir else subject_folder
            os.makedirs(save_dir, exist_ok=True)
            
            # Save reconstructed surface in MATLAB format (if requested)
            if save_surface:
                TessMat = {
                    'Vertices': reconstructed_merged_coords,
                    'Faces': full_faces + 1  # +1 because MATLAB starts at 1
                }
                try:
                    sio.savemat(os.path.join(save_dir, f'{subject_id}_reconstructed.mat'), {'TessMat': TessMat})
                except Exception as e:
                    print(f"Error while saving surface: {e}")
            
            # Save SH coefficients in pickle format
            try:
                with open(os.path.join(save_dir, f'{subject_id}_sh_coefficients.pkl'), 'wb') as f:
                    pickle.dump(sh_coefficients, f)
            except Exception as e:
                print(f"Error while saving coefficients: {e}")
            
            # Store results in the dictionary
            subject_results = {
                'lh_results': lh_results,
                'rh_results': rh_results,
                'merged_results': merged_results,
                'stats': stats,
                'sh_coefficients': sh_coefficients
            }
            
            results[subject_id] = subject_results
            
            # Indicate what was saved
            if save_surface:
                print(f"Reconstructed surface and SH coefficients (up to order {max_order_coeff}) saved for {subject_id}")
            else:
                print(f"SH coefficients (up to order {max_order_coeff}) saved for {subject_id}")
            
        except Exception as e:
            print(f"Error while processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nProcessing completed!")
    
    # Return the result of the specific subject if a subject_id was provided
    if len(subject_dirs) == 1 and os.path.basename(subject_dirs[0]) == os.path.basename(subject_id or ''):
        return results.get(os.path.basename(subject_dirs[0]))
    
    return results


def reconstruct_brain(lh_center=None, rh_center=None, coeffs_lh=None, coeffs_rh=None, 
                     Y_lh_full=None, Y_rh_full=None, tris=None, l=None, merge=True,
                     lh_vertices_file=None, lh_faces_file=None, rh_vertices_file=None, rh_faces_file=None, 
                     lh_resampled=None, rh_resampled=None, Y_lh=None, Y_rh=None, lmax=None, sigma=1e-4, 
                     lambda_reg=1e-8):
    """
    Versatile reconstruction function that supports different parameter styles.
    
    Original usage (from mesh_decimation.py):
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
    
    New usage:
        merged_results = reconstruct_brain(
            lh_vertices_file=lh_vertices_file,
            lh_faces_file=lh_faces_file,
            rh_vertices_file=rh_vertices_file,
            rh_faces_file=rh_faces_file,
            lh_resampled=lh_resampled,
            rh_resampled=rh_resampled,
            Y_lh=Y_lh,
            Y_rh=Y_rh,
            lmax=lmax,
            sigma=sigma,
            lambda_reg=lambda_reg
        )
    """
    # Determine which parameter style was used
    if lh_center is not None and coeffs_lh is not None:
        # Original style from mesh_decimation.py
        # Process left hemisphere
        Y_l = Y_lh_full[:, :(l+1)**2]
        org_coeffs_lh = coeffs_lh['organized_coeffs'] 
        coords_lh = SH.generate_surface(Y_l, l, 0, org_coeffs_lh)

        # Process right hemisphere
        Y_r = Y_rh_full[:, :(l+1)**2]
        org_coeffs_rh = coeffs_rh['organized_coeffs']
        coords_rh = SH.generate_surface(Y_r, l, 0, org_coeffs_rh)

        # Add centers
        lh_reconstruction = coords_lh + lh_center
        rh_reconstruction = coords_rh + rh_center

        if merge:
            # Merge hemispheres
            reconstructed_merged_coords, reconstructed_merged_tris = sp.merge_hemis(
                (lh_reconstruction, tris),
                (rh_reconstruction, tris)
            )
            return reconstructed_merged_coords, reconstructed_merged_tris
        else:
            # Return separate hemispheres
            return lh_reconstruction, rh_reconstruction, tris
    else:
        # New style
        # 1. Calculate coefficients for the left hemisphere
        lh_faces = load_faces(lh_faces_file)
        lh_vertices = load_vertices(lh_vertices_file)
        
        r_coords_lh, r_tris_lh = lh_resampled['coords'], lh_resampled['tris']
        lh_resampled_tuple = (r_coords_lh, r_tris_lh)
        
        lh_coeffs = SH.compute_coefficients_SVD(Y_lh, lh_resampled_tuple, lmax, lambda_reg)
        lh_reconstruction = SH.generate_surface(Y_lh, lmax, sigma, orders=lh_coeffs['organized_coeffs'])
        
        # 2. Calculate coefficients for the right hemisphere
        rh_faces = load_faces(rh_faces_file)
        rh_vertices = load_vertices(rh_vertices_file)
        
        r_coords_rh, r_tris_rh = rh_resampled['coords'], rh_resampled['tris']
        rh_resampled_tuple = (r_coords_rh, r_tris_rh)
        
        rh_coeffs = SH.compute_coefficients_SVD(Y_rh, rh_resampled_tuple, lmax, lambda_reg)
        rh_reconstruction = SH.generate_surface(Y_rh, lmax, sigma, orders=rh_coeffs['organized_coeffs'])
        
        # Create reconstruction results for both hemispheres
        lh_results = {
            'resampled_surface': lh_resampled_tuple,
            'reconstruction': lh_reconstruction,
            'coefficients': lh_coeffs,
            'original': {
                'vertices': lh_vertices,
                'faces': lh_faces
            }
        }
        
        rh_results = {
            'resampled_surface': rh_resampled_tuple,
            'reconstruction': rh_reconstruction,
            'coefficients': rh_coeffs,
            'original': {
                'vertices': rh_vertices,
                'faces': rh_faces
            }
        }
        
        # 3. Merge hemispheres
        lh_reconstruction_centered = lh_reconstruction + lh_resampled['center']
        rh_reconstruction_centered = rh_reconstruction + rh_resampled['center']
        
        reconstructed_merged_coords, reconstructed_merged_tris = sp.merge_hemis(
            (lh_reconstruction_centered, lh_results['resampled_surface'][1]),
            (rh_reconstruction_centered, rh_results['resampled_surface'][1])
        )
        
        merged_results = {
            'reconstructed': {
                'coords': reconstructed_merged_coords,
                'tris': reconstructed_merged_tris
            },
            'lh_results': lh_results,
            'rh_results': rh_results
        }
        
        return merged_results


if __name__ == "__main__":
    # Path configuration
    base_data_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation"
    data_path = os.path.join(base_data_path, "data")
    subjects_dir = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN"
    
    # Parameters
    sigma = 1e-4
    lmax = 30  # Maximum degree for surface reconstruction
    max_order_coeff = 80  # Maximum degree for coefficient storage
    lambda_reg = 0
    
    # Processing options
    process_all_subjects = True     # Process all subjects
    specific_subject = "sub-CC110033"  # Specific subject (used if process_all_subjects=False)
    visualize_results = False        # Visualize results
    save_reconstructed_surface = False  # Save reconstructed surface
    
    if process_all_subjects:
        print("Processing all subjects...")
        results = process_subjects(
            base_dir=subjects_dir,
            data_path=data_path,
            sigma=sigma,
            lmax=lmax,
            lambda_reg=lambda_reg,
            max_order_coeff=max_order_coeff,
            visualize_first=visualize_results,
            save_surface=save_reconstructed_surface
        )
    else:
        print(f"Processing specific subject: {specific_subject}")
        subject_result = process_subjects(
            base_dir=subjects_dir,
            data_path=data_path,
            subject_id=specific_subject,
            sigma=sigma,
            lmax=lmax,
            lambda_reg=lambda_reg,
            max_order_coeff=max_order_coeff,
            visualize_first=visualize_results,
            save_surface=save_reconstructed_surface
        )
        
        # We can do other specific processing on the result of a single subject if needed
        if subject_result:
            print(f"Successful processing of subject {specific_subject}")
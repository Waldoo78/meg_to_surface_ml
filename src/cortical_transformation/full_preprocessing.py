import os
import sys
import numpy as np
import pickle
import argparse
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface

# Utils imports
from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.cortical import surface_preprocess as sp
from utils.cortical import spherical_harmonics as SH


def create_fsaverage_template(output_dir, lmax, lambda_reg):
    """Create fsaverage template and compute its coefficients"""
    print("Creating fsaverage template...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load fsaverage
    fsaverage7 = fetch_surf_fsaverage(mesh='fsaverage7')
    surf_lh = surface.load_surf_mesh(fsaverage7['pial_left'])
    surf_rh = surface.load_surf_mesh(fsaverage7['pial_right'])
    
    # Process left hemisphere
    surface_mesh_lh = (surf_lh[0], surf_lh[1])
    coords_lh, tris_lh = sp.get_resampled_inner_surface(surface_mesh_lh, 'lh')
    center_lh = np.mean(coords_lh, axis=0)
    coords_lh = coords_lh - center_lh
    output_file_lh = os.path.join(output_dir, "lh_resampled.npz")
    np.savez(output_file_lh, coords=coords_lh, tris=tris_lh, center=center_lh)
    
    # Process right hemisphere
    surface_mesh_rh = (surf_rh[0], surf_rh[1])
    coords_rh, tris_rh = sp.get_resampled_inner_surface(surface_mesh_rh, 'rh')
    center_rh = np.mean(coords_rh, axis=0)
    coords_rh = coords_rh - center_rh
    output_file_rh = os.path.join(output_dir, "rh_resampled.npz")
    np.savez(output_file_rh, coords=coords_rh, tris=tris_rh, center=center_rh)
    
    print("Fsaverage template created successfully")
    return output_dir


def compute_fsaverage_coefficients(fsaverage_path, data_path, lmax, lambda_reg):
    """Compute spherical harmonic coefficients for fsaverage template"""
    print("Computing fsaverage coefficients...")
    
    # Load hemisphere-specific harmonics
    Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
    Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']
    
    # Slice according to lmax
    Y_lh = Y_lh_full[:, :(lmax+1)**2]
    Y_rh = Y_rh_full[:, :(lmax+1)**2]
    
    # Load resampled fsaverage data
    fsav_lh = np.load(os.path.join(fsaverage_path, "lh_resampled.npz"))
    fsav_rh = np.load(os.path.join(fsaverage_path, "rh_resampled.npz"))
    
    # Prepare resampled surfaces for coefficient computation
    resampled_lh = (fsav_lh['coords'], fsav_lh['tris'])
    resampled_rh = (fsav_rh['coords'], fsav_rh['tris'])
    
    # Compute coefficients for both hemispheres
    coeffs_fsav_lh = SH.compute_coefficients_SVD(Y_lh, resampled_lh, lmax, lambda_reg=lambda_reg)
    coeffs_fsav_rh = SH.compute_coefficients_SVD(Y_rh, resampled_rh, lmax, lambda_reg=lambda_reg)
    
    # Save coefficients
    with open(os.path.join(fsaverage_path, "coeffs_lh.pkl"), 'wb') as f:
        pickle.dump(coeffs_fsav_lh, f)
    with open(os.path.join(fsaverage_path, "coeffs_rh.pkl"), 'wb') as f:
        pickle.dump(coeffs_fsav_rh, f)
    
    print("Fsaverage coefficients computed and saved")


def process_all_subjects(main_folder, data_path, lmax, lambda_reg, subject_id=None):
    """Process all subjects: resample surfaces and compute coefficients"""
    print("Processing subjects...")
    
    # Load hemisphere-specific harmonics
    Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
    Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']
    
    # Slice according to lmax
    Y_lh = Y_lh_full[:, :(lmax+1)**2]
    Y_rh = Y_rh_full[:, :(lmax+1)**2]
    
    # Get list of subjects to process
    if subject_id:
        folders_to_process = [subject_id] if os.path.isdir(os.path.join(main_folder, subject_id)) else []
        if not folders_to_process:
            print(f"Subject {subject_id} not found!")
            return
    else:
        folders_to_process = [f for f in os.listdir(main_folder) 
                            if os.path.isdir(os.path.join(main_folder, f)) and f.startswith("sub-")]
    
    print(f"Found {len(folders_to_process)} subjects to process")
    
    for i, folder in enumerate(folders_to_process):
        folder_path = os.path.join(main_folder, folder)
        print(f"\nProcessing subject {i+1}/{len(folders_to_process)}: {folder}")
        
        try:
            # Step 1: Resample surfaces
            process_subject_resampling(folder_path)
            
            # Step 2: Compute coefficients
            process_subject_coefficients(folder_path, Y_lh, Y_rh, lmax, lambda_reg)
            
            # Step 3: Save centers
            save_subject_centers(folder_path)
            
            print(f"  Subject {folder} processed successfully")
            
        except Exception as e:
            print(f"  Error processing subject {folder}: {str(e)}")
            import traceback
            traceback.print_exc()


def process_subject_resampling(folder_path):
    """Resample surfaces for a single subject"""
    # Process left hemisphere
    try:
        left_vertices_file = os.path.join(folder_path, "lh_vertices.mat")
        left_faces_file = os.path.join(folder_path, "lh_faces.mat")
        output_file = os.path.join(folder_path, "lh_resampled.npz")
        
        left_faces = load_faces(left_faces_file)
        left_vertices = load_vertices(left_vertices_file)
        coords, tris = sp.get_resampled_inner_surface((left_vertices, left_faces), 'lh')
        center = np.mean(coords, axis=0)
        coords = coords - center
        np.savez(output_file, coords=coords, tris=tris, center=center)
        
    except Exception as e:
        print(f"    Error processing left hemisphere: {str(e)}")
        raise
    
    # Process right hemisphere
    try:
        right_vertices_file = os.path.join(folder_path, "rh_vertices.mat")
        right_faces_file = os.path.join(folder_path, "rh_faces.mat")
        output_file = os.path.join(folder_path, "rh_resampled.npz")
        
        right_faces = load_faces(right_faces_file)
        right_vertices = load_vertices(right_vertices_file)
        coords, tris = sp.get_resampled_inner_surface((right_vertices, right_faces), 'rh')
        center = np.mean(coords, axis=0)
        coords = coords - center
        np.savez(output_file, coords=coords, tris=tris, center=center)
        
    except Exception as e:
        print(f"    Error processing right hemisphere: {str(e)}")
        raise


def process_subject_coefficients(folder_path, Y_lh, Y_rh, lmax, lambda_reg):
    """Compute spherical harmonic coefficients for a single subject"""
    # Process left hemisphere
    coeffs_lh_path = os.path.join(folder_path, "coeffs_lh.pkl")
    left_resampled_data = np.load(os.path.join(folder_path, "lh_resampled.npz"))
    
    # Smooth the left hemisphere surface
    left_smoothed_coords = sp.smooth_surface(left_resampled_data['coords'], 
                                       left_resampled_data['tris'],
                                       n_iterations=5, 
                                       relaxation_factor=0.5)
    
    coeffs_lh = SH.compute_coefficients_SVD(Y_lh, 
                                          (left_smoothed_coords, left_resampled_data['tris']), 
                                          lmax, 
                                          lambda_reg)
    with open(coeffs_lh_path, 'wb') as f:
        pickle.dump(coeffs_lh, f)
    
    # Process right hemisphere
    coeffs_rh_path = os.path.join(folder_path, "coeffs_rh.pkl")
    right_resampled_data = np.load(os.path.join(folder_path, "rh_resampled.npz"))
    
    # Smooth the right hemisphere surface
    right_smoothed_coords = sp.smooth_surface(right_resampled_data['coords'], 
                                        right_resampled_data['tris'],
                                        n_iterations=5, 
                                        relaxation_factor=0.5)
    
    coeffs_rh = SH.compute_coefficients_SVD(Y_rh,
                                          (right_smoothed_coords, right_resampled_data['tris']),
                                          lmax, 
                                          lambda_reg)
    with open(coeffs_rh_path, 'wb') as f:
        pickle.dump(coeffs_rh, f)


def save_subject_centers(folder_path):
    """Save hemisphere centers for a single subject"""
    # Load centers from resampled files
    lh_data = np.load(os.path.join(folder_path, "lh_resampled.npz"))
    rh_data = np.load(os.path.join(folder_path, "rh_resampled.npz"))
    
    # Get centers
    lh_center = lh_data['center']
    rh_center = rh_data['center']
    
    # Save centers
    np.savez(os.path.join(folder_path, "lh_center.npz"), center=lh_center)
    np.savez(os.path.join(folder_path, "rh_center.npz"), center=rh_center)


def main():
    parser = argparse.ArgumentParser(description='Full preprocessing pipeline for cortical surfaces')
    parser.add_argument('--data_dir', required=True, help='Directory containing subject folders')
    parser.add_argument('--templates_dir', required=True, help='Directory containing Y_lh.npz and Y_rh.npz')
    parser.add_argument('--subject', help='Process only this specific subject')
    parser.add_argument('--lmax', type=int, default=80, help='Maximum spherical harmonic degree')
    parser.add_argument('--sigma', type=float, default=1e-7, help='Smoothing parameter')
    parser.add_argument('--lambda_reg', type=float, default=1e-7, help='Regularization parameter')
    parser.add_argument('--skip_fsaverage', action='store_true', help='Skip fsaverage template creation')
    
    args = parser.parse_args()
    
    # Create fsaverage output directory
    fsaverage_path = os.path.join(os.path.dirname(args.data_dir), "fsaverage")
    
    print("=== Full Preprocessing Pipeline ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Templates directory: {args.templates_dir}")
    print(f"Fsaverage output: {fsaverage_path}")
    print(f"lmax: {args.lmax}")
    print(f"Subject filter: {args.subject or 'All subjects'}")
    
    try:
        # Step 1: Create fsaverage template (if not skipped)
        if not args.skip_fsaverage:
            create_fsaverage_template(fsaverage_path, args.lmax, args.lambda_reg)
            compute_fsaverage_coefficients(fsaverage_path, args.templates_dir, args.lmax, args.lambda_reg)
        else:
            print("Skipping fsaverage template creation")
        
        # Step 2: Process all subjects
        process_all_subjects(args.data_dir, args.templates_dir, args.lmax, args.lambda_reg, args.subject)
        
        print("\n=== Preprocessing completed successfully! ===")
        print("Generated files for each subject:")
        print("  - lh_resampled.npz, rh_resampled.npz (resampled surfaces)")
        print("  - coeffs_lh.pkl, coeffs_rh.pkl (spherical harmonic coefficients)")
        print("  - lh_center.npz, rh_center.npz (hemisphere centers)")
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
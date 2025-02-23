import numpy as np
import pyvista as pv
import os 
from utils.file_manip.Matlab_to_array import load_faces, load_vertices
from utils.mathutils import compute_vertex_normals, build_template_adjacency_two_hemis,compute_mean_curvature,compute_curvature_differences,compute_hausdorff_metrics,compute_point_distances,compute_normal_differences
from utils.cortical import spherical_harmonics as SH 
from utils.cortical import surface_preprocess as sp
from utils.file_manip.vtk_processing import convert_triangles_to_pyvista

def reconstruct_hemisphere_core(vertices_file, faces_file, resampled_surface, Y, 
                              template_projection, lmax, sigma, lambda_reg=1e-8):
    """
    Core reconstruction function for one hemisphere.
    
    Args:
        vertices_file (str): Path to vertices file
        faces_file (str): Path to faces file
        resampled_surface (dict): Resampled surface data
        Y (ndarray): Spherical harmonics basis for this hemisphere
        template_projection (dict): Template data for this hemisphere
        lmax (int): Maximum degree for spherical harmonics
        sigma (float): Smoothing parameter
        lambda_reg (float): Regularization parameter
    
    Returns:
        dict: Reconstruction results
    """
    # Load and center original surface
    faces = load_faces(faces_file)
    vertices = load_vertices(vertices_file)
    vertices = vertices 

    # Extract resampled surface data
    r_coords, r_tris = resampled_surface['coords'], resampled_surface["tris"]
    resampled_surface_tuple = (r_coords, r_tris)
    
    # Compute coefficients and generate reconstructed surface
    coeffs = SH.compute_coefficients_SVD(Y, resampled_surface_tuple, lmax, lambda_reg)
    reconstruction_coords = SH.generate_surface(Y, lmax, sigma, orders=coeffs['organized_coeffs'])
    
    
    return {
        'resampled_surface': resampled_surface_tuple,
        'reconstruction': reconstruction_coords,
        'coefficients': coeffs,
        'original': {
            'vertices': vertices,
            'faces': faces
        }
    }

def merge_reconstructed_brain(lh_results, rh_results, lh_resampled, rh_resampled):
    """
    Merge the results of separately reconstructed brain hemispheres.
    
    Args:
        lh_results (dict): Results from left hemisphere reconstruction
        rh_results (dict): Results from right hemisphere reconstruction
        lh_resampled (dict): Left hemisphere resampled data
        rh_resampled (dict): Right hemisphere resampled data
    
    Returns:
        dict: Combined brain data
    """
    # 1. Original brain
    orig_merged_coords, orig_merged_tris = sp.merge_hemis(
        (lh_results['original']['vertices'], lh_results['original']['faces']),
        (rh_results['original']['vertices'], rh_results['original']['faces'])
    )

    # 2. Resampled brain
    lh_resampled_coords = lh_resampled['coords'] + lh_resampled['center']
    rh_resampled_coords = rh_resampled['coords'] + rh_resampled['center']
    resampled_merged_coords, resampled_merged_tris = sp.merge_hemis(
        (lh_resampled_coords, lh_resampled['tris']),
        (rh_resampled_coords, rh_resampled['tris'])
    )
    
    # 3. Reconstructed brain
    lh_reconstruction = lh_results['reconstruction'] + lh_resampled['center']
    rh_reconstruction = rh_results['reconstruction'] + rh_resampled['center']
    reconstructed_merged_coords, reconstructed_merged_tris = sp.merge_hemis(
        (lh_reconstruction, lh_results['resampled_surface'][1]),
        (rh_reconstruction, rh_results['resampled_surface'][1])
    )

    return {
        'original': {
            'coords': orig_merged_coords,
            'tris': orig_merged_tris
        },
        'resampled': {
            'coords': resampled_merged_coords,
            'tris': resampled_merged_tris
        },
        'reconstructed': {
            'coords': reconstructed_merged_coords,
            'tris': reconstructed_merged_tris
        },
        'lh_results': lh_results,
        'rh_results': rh_results
    }

def visualize_hemisphere(hemisphere_results, title_prefix="", highlight_indices=None):
    """
    Visualization function for one hemisphere.
    
    Args:
        hemisphere_results (dict): Results from hemisphere reconstruction
        title_prefix (str): Prefix for subplot titles (e.g., "Left" or "Right")
        highlight_indices (list): Optional indices of vertices to highlight
    """
    vertices = hemisphere_results['original']['vertices']
    faces = hemisphere_results['original']['faces']
    r_coords, r_tris = hemisphere_results['resampled_surface']
    reconstruction_coords = hemisphere_results['reconstruction']
    
    p = pv.Plotter(shape=(1, 3))
    
    # Original Surface
    p.subplot(0, 0)
    mesh1 = pv.PolyData(vertices-np.mean(vertices, axis=0), convert_triangles_to_pyvista(faces))
    p.add_mesh(mesh1, show_edges=True, smooth_shading=True, color="blue")
    p.add_text(f"{title_prefix} Original", position='upper_edge', font_size=10)

    # Resampled Surface
    p.subplot(0, 1)
    mesh2 = pv.PolyData(r_coords, convert_triangles_to_pyvista(r_tris))
    p.add_mesh(mesh2, show_edges=True, smooth_shading=True, color="green")
    p.add_text(f"{title_prefix} Resampled", position='upper_edge', font_size=10)

    # Reconstructed Surface
    p.subplot(0, 2)
    mesh3 = pv.PolyData(reconstruction_coords, convert_triangles_to_pyvista(r_tris))
    p.add_mesh(mesh3, show_edges=True, smooth_shading=True, color="red")
    p.add_text(f"{title_prefix} Reconstructed", position='upper_edge', font_size=10)

    p.link_views()
    p.show()

def visualize_merged_brain(merged_results):
    """
    Visualize the merged brain results in a three-panel view.
    
    Args:
        merged_results (dict): Results from merge_reconstructed_brain
    """
    p = pv.Plotter(shape=(1, 3))
    
    # Original brain
    tris_orig = convert_triangles_to_pyvista(merged_results['original']['tris'])
    mesh_orig = pv.PolyData(merged_results['original']['coords'], tris_orig)
    p.subplot(0, 0)
    p.add_mesh(mesh_orig, show_edges=True, smooth_shading=True, color="white")
    p.add_text("Original Brain", position='upper_edge', font_size=10)
    
    # Resampled brain
    tris_resampled = convert_triangles_to_pyvista(merged_results['resampled']['tris'])
    mesh_resampled = pv.PolyData(merged_results['resampled']['coords'], tris_resampled)
    p.subplot(0, 1)
    p.add_mesh(mesh_resampled, show_edges=True, smooth_shading=True, color="lightblue")
    p.add_text("Resampled Brain", position='upper_edge', font_size=10)
    
    # Reconstructed brain
    tris_reconstructed = convert_triangles_to_pyvista(merged_results['reconstructed']['tris'])
    mesh_reconstructed = pv.PolyData(merged_results['reconstructed']['coords'], tris_reconstructed)
    p.subplot(0, 2)
    p.add_mesh(mesh_reconstructed, show_edges=True, smooth_shading=True, color="lightgreen")
    p.add_text("Reconstructed Brain", position='upper_edge', font_size=10)
    
    p.link_views()
    p.camera_position = 'xz'
    p.show()

def reconstruct_brain(lh_center, rh_center, coeffs_lh, coeffs_rh, Y_lh_full, Y_rh_full, tris, l, merge):
    # Process left hemisphere
    Y_l = Y_lh_full[:, :(l+1)**2]
    org_coeffs_lh = coeffs_lh['organized_coeffs'] 
    coords_lh = SH.generate_surface(Y_l, l, 0, org_coeffs_lh)

    # Process right hemisphere  
    Y_r = Y_rh_full[:, :(l+1)**2]
    org_coeffs_rh = coeffs_rh['organized_coeffs']
    coords_rh = SH.generate_surface(Y_r, l, 0, org_coeffs_rh)

    # Add centers back
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

if __name__ == "__main__":
   # Parameters
   sigma = 1e-4
   lmax = 30
   lambda_reg = 0

   # Path configuration
   base_data_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation"
   data_path = os.path.join(base_data_path, "data")
   subject_folder = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN\sub-CC721648"

   # Load hemisphere-specific templates
   template_projection_lh = np.load(os.path.join(data_path, "lh_sphere_projection.npz"))
   template_projection_rh = np.load(os.path.join(data_path, "rh_sphere_projection.npz"))

   # Load separate harmonics and slice according to lmax
   Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
   Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']
   
   # Left hemisphere uses first part, right hemisphere uses second part
   Y_lh = Y_lh_full[:, :(lmax+1)**2]
   Y_rh = Y_rh_full[:, :(lmax+1)**2:]

   # Load resampled data for both hemispheres
   lh_resampled = np.load(os.path.join(subject_folder, "lh_resampled.npz"))
   rh_resampled = np.load(os.path.join(subject_folder, "rh_resampled.npz"))

   print("Starting left hemisphere reconstruction...")
   lh_results = reconstruct_hemisphere_core(
       vertices_file=os.path.join(subject_folder, "lh_vertices.mat"),
       faces_file=os.path.join(subject_folder, "lh_faces.mat"),
       resampled_surface=lh_resampled,
       Y=Y_lh,
       template_projection=template_projection_lh,
       lmax=lmax,
       sigma=sigma,
       lambda_reg=lambda_reg
   )

   print("Starting right hemisphere reconstruction...")
   rh_results = reconstruct_hemisphere_core(
       vertices_file=os.path.join(subject_folder, "rh_vertices.mat"),
       faces_file=os.path.join(subject_folder, "rh_faces.mat"),
       resampled_surface=rh_resampled,
       Y=Y_rh,
       template_projection=template_projection_rh,
       lmax=lmax,
       sigma=sigma,
       lambda_reg=lambda_reg
   )

   char_size = np.max(np.ptp(lh_results['resampled_surface'][0], axis=0))
   print(f'Characteristic size: {char_size:.5f}')

   # Visualize individual hemispheres if needed
   print("Visualizing individual hemispheres...")
   visualize_hemisphere(lh_results, title_prefix="Left Hemisphere")
   visualize_hemisphere(rh_results, title_prefix="Right Hemisphere")

   # Merge hemispheres
   print("Merging hemispheres...")
   merged_results = merge_reconstructed_brain(lh_results, rh_results, lh_resampled, rh_resampled)

   # Visualize merged brain
   print("Visualizing merged brain...")
   visualize_merged_brain(merged_results)

   # Error Analysis
   print("Performing error analysis...")
   vertices_resampled = merged_results['resampled']['coords']
   vertices_reconstructed = merged_results['reconstructed']['coords']
   
   # Create faces for full brain
   n_vertices_lh = len(template_projection_lh['sphere_coords'])
   rh_tris_adjusted = template_projection_rh['sphere_tris'] + n_vertices_lh
   full_faces = np.vstack([
       template_projection_lh['sphere_tris'],
       rh_tris_adjusted
   ])



#    TessMat = {
#         'Vertices': vertices_reconstructed,
#         'Faces': full_faces + 1  # +1 as Matlab begins at 1 
#     }
#    sio.savemat(os.path.join(subject_folder, 'brain_reconstructed.mat'), {'TessMat': TessMat})

   
   vertex_to_faces = build_template_adjacency_two_hemis(
       template_projection_lh['sphere_tris'],
       template_projection_rh['sphere_tris']
   )

   # Smooth resampled surface
   print("Smoothing surfaces and computing normals...")
   vertices_resampled_smooth = sp.smooth_surface(vertices_resampled, full_faces, 
                                            n_iterations=5, relaxation_factor=0.5)
   
   # Point-to-point distance visualization
   print("\nVisualizing point-to-point distances...")
   pl = pv.Plotter(shape=(1, 2))
   
   # Resampled surface
   pl.subplot(0, 0)
   mesh_resampled = pv.PolyData(vertices_resampled_smooth, convert_triangles_to_pyvista(full_faces))
   pl.add_mesh(mesh_resampled, color='lightgray', show_edges=True, 
               edge_color='black', line_width=1)
   pl.add_text("Resampled Surface", position='upper_edge')
   pl.view_isometric()
   
   # Distance errors
   pl.subplot(0, 1)
   mesh_reconstructed = pv.PolyData(vertices_reconstructed, convert_triangles_to_pyvista(full_faces))
   distances = np.sqrt(np.sum((vertices_resampled_smooth - vertices_reconstructed)**2, axis=1)) * 1000  # Convert to mm
   
   pl.add_mesh(mesh_reconstructed, scalars=distances, cmap='viridis',
               show_edges=True, edge_color='black', line_width=1,
               scalar_bar_args={'title': 'Distance Error (mm)',
                              'n_labels': 5,
                              'position_x': 0.05,
                              'position_y': 0.05,
                              'width': 0.3,
                              'height': 0.05})
   pl.add_text(f"Point-to-point distance errors\nMean: {np.mean(distances):.2f} mm\nMax: {np.max(distances):.2f} mm", 
               position='upper_edge')
   pl.view_isometric()
   pl.link_views()
   pl.show()

   # Compute normals and analyze differences
   print("\nComputing normal analysis...")
   normals_resampled = compute_vertex_normals(vertices_resampled_smooth, full_faces, vertex_to_faces, n_rings=2)
   normals_reconstructed = compute_vertex_normals(vertices_reconstructed, full_faces, vertex_to_faces, n_rings=2)
   angles = np.degrees(np.arccos(np.clip(np.sum(normals_resampled * normals_reconstructed, axis=1), -1.0, 1.0)))
   
   # Normal angle error visualization
   print("\nVisualizing normal angle errors...")
   pl = pv.Plotter(shape=(1, 3))
   
   # Resampled surface
   pl.subplot(0, 0)
   mesh_resampled = pv.PolyData(vertices_resampled_smooth, convert_triangles_to_pyvista(full_faces))
   pl.add_mesh(mesh_resampled, color='lightgray', show_edges=True, 
               edge_color='black', line_width=1)
   pl.add_text("Resampled Surface", position='upper_edge')
   pl.view_isometric()
   
   # Reconstructed surface
   pl.subplot(0, 1)
   mesh_reconstructed = pv.PolyData(vertices_reconstructed, convert_triangles_to_pyvista(full_faces))
   pl.add_mesh(mesh_reconstructed, color='lightgray', show_edges=True, 
               edge_color='black', line_width=1)
   pl.add_text("Reconstructed Surface", position='upper_edge')
   pl.view_isometric()
   
   # Normal angle errors with better visibility
   pl.subplot(0, 2)
   high_angle_mask = angles >= 45
   angle_colors = np.zeros((len(vertices_reconstructed), 3))
   angle_colors[~high_angle_mask] = np.array([0.8, 0.8, 0.8])  # light gray
   angle_colors[high_angle_mask] = np.array([1.0, 0.0, 0.0])   # bright red
   
   pl.add_mesh(mesh_reconstructed, 
               scalars=angle_colors,
               rgb=True,
               show_edges=True, 
               edge_color='black', 
               line_width=1)
   pl.add_text(f"Normal angle errors\nMean: {np.mean(angles):.2f}°\nMax: {np.max(angles):.2f}°\nRed areas: >45°\nPercentage red: {100*np.sum(high_angle_mask)/len(angles):.1f}%", 
               position='upper_edge')
   pl.view_isometric()
   pl.link_views()
   pl.show()

   # Curvature analysis
   print("\nComputing curvature analysis...")
   curvature_resampled = compute_mean_curvature(vertices_resampled_smooth, full_faces, vertex_to_faces)
   curvature_reconstructed = compute_mean_curvature(vertices_reconstructed, full_faces, vertex_to_faces)
   curvature_diff = np.abs(curvature_resampled - curvature_reconstructed)


   # Curvature visualization
   print("\nVisualizing mean curvature...")
   pl = pv.Plotter(shape=(1, 2))

   # Resampled surface curvature
   pl.subplot(0, 0)
   curvature_range = np.maximum(abs(np.min(curvature_resampled)), abs(np.max(curvature_resampled)))
   pl.add_mesh(mesh_resampled, scalars=curvature_resampled, cmap='coolwarm',
               show_edges=True, edge_color='black', line_width=1,
               clim=[-curvature_range, curvature_range],
               scalar_bar_args={'title': 'Mean Curvature (1/mm)',
                              'n_labels': 5,
                              'position_x': 0.05,
                              'position_y': 0.05,
                              'width': 0.3,
                              'height': 0.05})
   pl.add_text(f"Resampled Surface Mean Curvature\nRange: [{np.min(curvature_resampled):.3f}, {np.max(curvature_resampled):.3f}]", 
               position='upper_edge')
   pl.view_isometric()

   # Curvature differences
   pl.subplot(0, 1)
   pl.add_mesh(mesh_reconstructed, scalars=curvature_diff, cmap='viridis',
               show_edges=True, edge_color='black', line_width=1,
               scalar_bar_args={'title': 'Curvature Difference (1/mm)',
                              'n_labels': 5,
                              'position_x': 0.05,
                              'position_y': 0.05,
                              'width': 0.3,
                              'height': 0.05})
   pl.add_text(f"Curvature Error\nMean: {np.mean(curvature_diff):.3f}\nMax: {np.max(curvature_diff):.3f}", 
               position='upper_edge')
   pl.view_isometric()

   pl.link_views()
   pl.show()

   # Print summary statistics
   print("\n=== Analysis Summary ===")
   print("\nPoint-to-point distances (mm):")
   print(f"Mean: {np.mean(distances):.2f}")
   print(f"Max: {np.max(distances):.2f}")
   print(f"95th percentile: {np.percentile(distances, 95):.2f}")

   print("\nNormal angles (degrees):")
   print(f"Mean: {np.mean(angles):.2f}")
   print(f"Max: {np.max(angles):.2f}")
   print(f"95th percentile: {np.percentile(angles, 95):.2f}")
   print(f"Percentage >45°: {100*np.sum(angles > 45)/len(angles):.1f}%")

   print("\n=== Mean Curvature Analysis ===")
   print("\nResampled surface:")
   print(f"- Mean curvature: {np.mean(curvature_resampled):.4f}")
   print(f"- Standard deviation: {np.std(curvature_resampled):.4f}")
   print(f"- Min/Max: {np.min(curvature_resampled):.4f} / {np.max(curvature_resampled):.4f}")

   print("\nReconstructed surface:")
   print(f"- Mean curvature: {np.mean(curvature_reconstructed):.4f}")
   print(f"- Standard deviation: {np.std(curvature_reconstructed):.4f}")
   print(f"- Min/Max: {np.min(curvature_reconstructed):.4f} / {np.max(curvature_reconstructed):.4f}")

   print("\nCurvature differences:")
   print(f"- Mean absolute difference: {np.mean(curvature_diff):.4f}")
   print(f"- Difference std: {np.std(curvature_diff):.4f}")
   print(f"- Max difference: {np.max(curvature_diff):.4f}")
   print(f"- 75th percentile: {np.percentile(curvature_diff, 75):.4f}")
   print(f"- 95th percentile: {np.percentile(curvature_diff, 95):.4f}")
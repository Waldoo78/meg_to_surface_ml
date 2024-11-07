import os
import tempfile
import numpy as np
import pyvista as pv
from utils.file_manip import vtk_processing
from utils.file_manip import Matlab_to_array 
from nilearn import surface

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
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "mesh", "S3MAP-main", "s3all.py")
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
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "mesh", "S3MAP-main", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True --device CPU"
    print(f"Executing: {cmd}")
    os.system(cmd)

    # Load results
    if not os.path.exists(target_vtk):
        print(f"Looking for file: {target_vtk}")
        raise FileNotFoundError("S3MAP inner surface output file not found")

    target_coords, target_triangles = vtk_processing.vtk_mesh_to_array(target_vtk)
    
    return target_coords, target_triangles

def main_test():
    from nilearn.datasets import fetch_surf_fsaverage
    def extract_hemisphere(vertices, faces, hemisphere='left'):
        """
        Extrait un hémisphère avec marge négative pour coupe nette
        """
        # Trouver le centre sur l'axe y
        center_y = np.mean(vertices[:, 1])
        
        # Ajouter une marge négative pour éviter le débordement
        # 5% de l'écart-type comme marge
        
        # Séparer selon y avec marge négative
        if hemisphere == 'left':
            mask = vertices[:, 1] <= (center_y )  # On coupe un peu plus à gauche
        else:
            mask = vertices[:, 1] > (center_y)  # On coupe un peu plus à droite
            
        # Extraire les vertices
        hemisphere_vertices = vertices[mask]
        
        # Mettre à jour les faces
        vertex_map = np.cumsum(mask) - 1
        hemisphere_faces = []
        for face in faces:
            if all(mask[face]):
                new_face = vertex_map[face]
                hemisphere_faces.append(new_face)
                
        return hemisphere_vertices, np.array(hemisphere_faces)

    def visualize_mesh(vertices, faces, title):
        """Visualise le maillage"""
        plotter = pv.Plotter()
        mesh = pv.PolyData(vertices, np.hstack((np.full((len(faces), 1), 3), faces)))
        plotter.add_mesh(mesh, show_edges=True, color='white', edge_color='black')
        plotter.add_text(title)
        plotter.show()

    # Charger les données
    print("Chargement des données...")
    faces = Matlab_to_array.load_faces(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\Faces.mat")
    vertices = Matlab_to_array.load_vertices(r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\data\Vertices.mat")
    
    # Extraire l'hémisphère gauche
    print("\nExtraction de l'hémisphère gauche...")
    lh_vertices, lh_faces = extract_hemisphere(vertices, faces, 'left')
    
    # Visualiser le résultat
    visualize_mesh(vertices, faces, "Cerveau complet")
    visualize_mesh(lh_vertices, lh_faces, f"Hémisphère gauche ({len(lh_vertices)} vertices)")
    
    # Statistiques
    print(f"\nNombre de vertices dans l'hémisphère : {len(lh_vertices)}")
    print(f"Nombre de faces dans l'hémisphère : {len(lh_faces)}")
    print(f"Ratio de vertices (devrait être proche de 0.5) : {len(lh_vertices)/len(vertices):.3f}")


    try:
        print("\nPerforming spherical projection...")
        lh_sphere_coords, lh_sphere_tris = get_spherical_projection((lh_vertices, lh_faces), hemisphere='lh')
        
        print("Performing surface resampling...")
        lh_40k_coords, lh_40k_tris = get_resampled_inner_surface((lh_vertices, lh_faces), hemisphere='lh')
        
        # Visualize results
        visualize_mesh(lh_40k_coords, lh_40k_tris, "Resampled mesh (40962 vertices)")
        visualize_mesh(lh_sphere_coords, lh_sphere_tris, "Spherical projection (40962 vertices)")
        
        # Print summary
        print("\nMesh sizes:")
        print(f"Original hemisphere: {len(lh_vertices)} vertices")
        print(f"Resampled mesh: {len(lh_40k_coords)} vertices")
        print(f"Spherical projection: {len(lh_sphere_coords)} vertices")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")


def main_freesurfer():
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn import surface
    import numpy as np
    import pyvista as pv
    
    def visualize_mesh(vertices, faces, title):
        """Visualise le maillage"""
        plotter = pv.Plotter()
        mesh = pv.PolyData(vertices, np.hstack((np.full((len(faces), 1), 3), faces)))
        plotter.add_mesh(mesh, show_edges=True, color='white', edge_color='black')
        plotter.add_text(title)
        plotter.show()
    
    def compute_vertex_error(vertices1, vertices2):
        """
        Calcule l'erreur moyenne entre les positions des sommets correspondants
        """
        # Calcul des distances euclidiennes entre les sommets correspondants
        distances = np.sqrt(np.sum((vertices1 - vertices2) ** 2, axis=1))
        
        # Statistiques sur les erreurs
        mean_error = np.mean(distances)
        std_error = np.std(distances)
        max_error = np.max(distances)
        
        return mean_error, std_error, max_error

    try:
        print("Téléchargement de fsaverage6...")
        fsaverage = fetch_surf_fsaverage(mesh='fsaverage6')
        
        # Récupération de l'hémisphère gauche uniquement
        vertices, faces = surface.load_surf_mesh(fsaverage['pial_left'])
        
        print(f"Surface fsaverage6: {len(vertices)} vertices")
        
        # Obtention de la surface rééchantillonnée
        print("\nRééchantillonnage de la surface...")
        resampled_coords, resampled_tris = get_resampled_inner_surface((vertices, faces), hemisphere='lh')
        
        # Calcul des erreurs entre les sommets correspondants
        if len(vertices) == len(resampled_coords):
            mean_error, std_error, max_error = compute_vertex_error(vertices, resampled_coords)
            print("\nStatistiques d'erreur entre les sommets:")
            print(f"Erreur moyenne: {mean_error:.4f} unités")
            print(f"Écart-type de l'erreur: {std_error:.4f} unités")
            print(f"Erreur maximale: {max_error:.4f} unités")
        else:
            print(f"\nAttention: Nombre de sommets différent entre les surfaces")
            print(f"fsaverage6: {len(vertices)} vertices")
            print(f"Rééchantillonnée: {len(resampled_coords)} vertices")
        
        # Visualisation des deux surfaces
        visualize_mesh(vertices, faces, "Surface fsaverage6 originale")
        visualize_mesh(resampled_coords, resampled_tris, "Surface rééchantillonnée")
            
    except Exception as e:
        print(f"\nErreur lors du traitement: {str(e)}")

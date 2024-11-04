import numpy as np
import scipy.io
import os

def load_faces(file_path):
    """Charge les faces depuis un fichier .mat"""
    mat_contents = scipy.io.loadmat(file_path)
    faces = mat_contents['Faces']  # Utiliser directement la clé 'Faces'
    return faces.astype(np.int32) - 1  # Convertir en 0-based indexing

def load_vertices(file_path):
    """Charge les vertices depuis un fichier .mat"""
    mat_contents = scipy.io.loadmat(file_path)
    vertices = mat_contents['Vertices']  # Utiliser directement la clé 'Vertices'
    return vertices.astype(np.float64)

if __name__ == "__main__":
    try:
        # Chemins des fichiers
        file_faces = r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Faces_1.mat"
        file_vertices = r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Vertices_1.mat"
        
        # Charger les faces
        faces = load_faces(file_faces)
        print(f"Faces chargées : shape={faces.shape}, type={faces.dtype}")
        print("Premier triangle :", faces[0])
        
        # Charger les vertices
        vertices = load_vertices(file_vertices)
        print(f"\nVertices chargés : shape={vertices.shape}, type={vertices.dtype}")
        print("Premier point :", vertices[0])
        
        # Vérifications
        print("\nVérifications :")
        print(f"Nombre de triangles : {len(faces)}")
        print(f"Nombre de points : {len(vertices)}")
        print(f"Index maximum dans faces : {faces.max()}")
        print(f"Index minimum dans faces : {faces.min()}")
        
    except Exception as e:
        print(f"Erreur : {str(e)}")
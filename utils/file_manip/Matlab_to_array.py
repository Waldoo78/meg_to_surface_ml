import numpy as np
import scipy.io

def load_faces(file_path):
    """Charge les faces depuis un fichier .mat"""
    mat_contents = scipy.io.loadmat(file_path)
    # Récupérer la première clé qui n'est pas '__'
    key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
    faces = mat_contents[key]
    return faces.astype(np.int32) - 1

def load_vertices(file_path):
    """Charge les vertices depuis un fichier .mat"""
    mat_contents = scipy.io.loadmat(file_path)
    key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
    vertices = mat_contents[key]
    return vertices.astype(np.float64)




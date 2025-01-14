import pyvista as pv
import numpy as np
from vtk import vtkPoints, vtkCellArray, vtkTriangle, vtkPolyData, vtkPolyDataWriter


def vtk_mesh_to_array(vtk_file):
    
    mesh = pv.read(vtk_file)
    points=np.array(mesh.points)
    
    faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]

    return points, faces

def save_to_vtk(coords,triangles, output_file):
       
       # Create VTK points
       points = vtkPoints()
       for coord in coords:
           points.InsertNextPoint(coord[0], coord[1], coord[2])
       
       # Create cells (triangles) 
       cells = vtkCellArray()
       for triangle in triangles:
           vtk_triangle = vtkTriangle()
           for i in range(3):
               vtk_triangle.GetPointIds().SetId(i, triangle[i])
           cells.InsertNextCell(vtk_triangle)
       
       # Create polydata
       polydata = vtkPolyData()
       polydata.SetPoints(points)
       polydata.SetPolys(cells)
       
       # Write file
       writer = vtkPolyDataWriter()
       writer.SetFileName(output_file)
       writer.SetInputData(polydata)
       writer.Write()

def convert_triangles_to_pyvista(triangles):
    return triangles if triangles.shape[1] == 4 else np.column_stack((np.full(len(triangles), 3), triangles))


if __name__=="__main__":
     from utils.mathutils import cart_to_sph
     import pickle
     vtk_file=r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\cortical_transfo_data_utils\sphere_163842_rotated_0.vtk"
     points, faces=vtk_mesh_to_array(vtk_file)
     _,theta,phi=cart_to_sph(points)
     template_projection={}
     template_projection_path=r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\cortical_transfo_data_utils\template_projection"
     template_projection['theta']=theta
     template_projection["phi"]=phi
     template_projection['sphere_tris']=faces
     template_projection['coords']=points 
     with open(template_projection_path, "wb") as f:
          pickle.dump(template_projection,f)

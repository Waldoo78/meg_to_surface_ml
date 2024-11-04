import pyvista as pv
import numpy as np
from vtk import vtkPoints, vtkCellArray, vtkTriangle, vtkPolyData, vtkPolyDataWriter


def vtk_mesh_to_array(vtk_file):
    
    # Charger les maillages VTK
    mesh = pv.read(vtk_file)
    points=np.array(mesh.points)
    
    faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]

    return points, faces

def save_to_vtk(coords,triangles, output_file):
       """
       Save surface as VTK format
       
       Args:
           output_file (str): Path to VTK file to create
       """
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
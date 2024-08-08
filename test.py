from mesher import RectangleMesh
from FEmethod import FEM

# initialise mesh object
mesh = RectangleMesh(length=1,width=1,nx=10,ny=10)
# create mesh
mesh.create_rectangular_mesh()
# create mesh plot
mesh.plot_mesh()


FE = FEM(nodes=mesh.nodes,elements=mesh.elements)
FE.B_matrix(element=0)

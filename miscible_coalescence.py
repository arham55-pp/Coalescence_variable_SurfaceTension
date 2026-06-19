from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import degree
import os
import numpy as np

from pyoomph.equations.navier_stokes import NavierStokesSlipLength, NavierStokesContactAngle, StokesEquations, NavierStokesEquations, NavierStokesFreeSurface
from pyoomph.equations.ALE import PseudoElasticMesh,HyperelasticSmoothedMesh
from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations

from pyoomph.utils.dropgeom import DropletGeometry
from pyoomph.meshes.remesher import Remesher2d

from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from pyoomph.output.plotting import MatplotlibPlotter
#PLOTTING C AND U ONLY
class DropletPlotter(MatplotlibPlotter):
    def define_plot(self):
        self.set_view(-1.5,-0.1,1.5,0.9)
        cb_c=self.add_colorbar("c",position="top left",cmap="coolwarm")
        cb_v=self.add_colorbar("velocity",position="top right",cmap="viridis")
        self.add_plot("liquid/c",colorbar=cb_c)
        self.add_plot("liquid/velocity",colorbar=cb_v,mode="streamlines")
        self.add_plot("liquid/interface")
        self.add_plot("liquid/substrate")
        self.add_time_label("top center")
        

# We need more elements (finer resolution where h is small). However, we only want to have it in the center near x=0, not at the outer contact lines
# This blends from 1 (coarse) to h(x) in the center
def get_relative_mesh_size_from_x_and_h(x,hfunc):
    xabs=numpy.abs(x)
    blend_to_coarse=numpy.arctan(xabs)/pi # Blend to that at x~1 (apex of the droplets), we go to coarse resolution
    return blend_to_coarse+(1-blend_to_coarse)*hfunc(x) # Otherwise, near the connection point, we take the local height 

# When the mesh is too deformed, we have to make a new mesh. Here, we must extract the old height profile h(x) to use it for the mesh size of the new mesh
class TwoDropletRemesherWithHeightDependentSize(Remesher2d):
    def remesh(self):
        problem=self.template.get_problem()
        data=problem.get_cached_mesh_data("liquid/interface",nondimensional=True)        
        segs=data.get_interface_line_segments()[0][0] # Only one line segment for the interface
        x=data.get_coordinates()[0,:][segs]
        h=data.get_coordinates()[1,:][segs]
        if x[0]>x[-1]:
            x=x[::-1]
            h=h[::-1]
        # Now we have x and h as point list (ordered by ascending x)
        # Interpolated function
        hinter=interp1d(x,h,kind="cubic",fill_value="extrapolate")
        # Set the mesh size by a callback function that depends on x and h(x)
        self._mesh_size_callback=lambda dim,tag,x,y,z,lc: problem.mesh_size*get_relative_mesh_size_from_x_and_h(x,hinter)
        # And do the remeshing (implemented in the base class)
        return super().remesh()

# Initial mesh 
class SlightlyConnectedTwoDropletMesh(GmshTemplate):
    def define_geometry(self):
        self.mesh_mode="only_quads"
        pr=cast("NavierStokesCoalescenceProblem",self.get_problem())
        if float(90*degree-pr.theta)<1e-5:
            raise ValueError("theta must be lower than 90 degrees.")
        # Utility to calculate spherical cap quantities
        geom=DropletGeometry(base_radius=1,contact_angle=pr.theta)        
        
        d=pr.initial_overlap
        def h_drop(x):
            r=1-d/2-absolute(x)
            return -geom.curv_radius+geom.apex_height+square_root(geom.curv_radius**2-r**2)        

        # sample the shape in x
        samples=numpy.linspace(-2+d/2,2-d/2,1000,endpoint=True)        
        pts=[self.point(x,h_drop(x)) for x in samples] # add points for the interface
        self.spline(pts,name="interface") # and connect them with a spline curve
        self.line(pts[0],pts[-1],name="substrate") # substrate line
        # and plane surface
        self.plane_surface("substrate","interface",name="liquid")
        # Use the height dependent mesh size callback for the remeshing (initially, we can just use the initial shape)
        self._mesh_size_callback=lambda dim,tag,x,y,z,lc: pr.mesh_size*get_relative_mesh_size_from_x_and_h(x,h_drop)
        # And attach our custom remesher that extracts the height profile from the old mesh and uses it for the new mesh size
        self.remesher=TwoDropletRemesherWithHeightDependentSize(self)
        
def find_neck(x, h):
    """Find coalescence neck (local minimum closest to x=0).

    Uses scipy.signal.find_peaks with prominence filtering to robustly
    identify local minima, then selects the one closest to x=0.

    Args:
        x: position array
        h: height array

    Returns:
        x0: neck position
        h0: neck height
    """
    sort_idx = np.argsort(x)
    x_sorted, h_sorted = x[sort_idx], h[sort_idx]

    # Find local minima (peaks in -h) with prominence filtering
    prominence = 0.01 * (h_sorted.max() - h_sorted.min())
    min_indices, _ = find_peaks(-h_sorted, prominence=prominence)

    if len(min_indices) == 0:
        # Fallback: interpolate at x=0
        return 0.0, np.interp(0.0, x_sorted, h_sorted)

    # Select minimum closest to x=0
    min_positions = x_sorted[min_indices]
    closest_idx = min_indices[np.argmin(np.abs(min_positions))]

    return x_sorted[closest_idx], h_sorted[closest_idx]


class NeckLogger(GenericProblemHooks):
    def __init__(self, filename="neck_dynamics.csv", bridge_half_width=0.25):
        super().__init__()
        self.filename = filename
        self.bridge_half_width = bridge_half_width

    def actions_after_initialise(self):
        problem = self.get_problem()
        os.makedirs(problem.get_output_directory(), exist_ok=True)
        with open(problem.get_output_directory(self.filename), "w") as f:
            f.write("step,time,x0,h0\n")

    def actions_on_output(self, outstep):
        problem = self.get_problem()
        data = problem.get_cached_mesh_data("liquid/interface", nondimensional=True)
        segs = data.get_interface_line_segments()[0][0]
        x = data.get_coordinates()[0, :][segs]
        h = data.get_coordinates()[1, :][segs]
        bridge = numpy.abs(x) <= self.bridge_half_width
        if numpy.count_nonzero(bridge) >= 3:
            x = x[bridge]
            h = h[bridge]
        x0, h0 = find_neck(x, h)

        with open(problem.get_output_directory(self.filename), "a") as f:
            f.write(f"{outstep},{float(problem.get_current_time())},{x0},{h0}\n")
     
        
# Problem class: Here, we set up the problem        
class NavierStokesCoalescenceProblem(Problem):
    def __init__(self):
        super().__init__()                
        
        self.theta=30*degree
        self.slip=0.00001 # Slip length (since we do not use a precursor film, we need some slip to avoid singularity at the contact line)
        self.Pe=50
        self.Re=0
        self.beta=0.5
        self.mesh_size=0.2 # The smaller, the finer the mesh    
        self.initial_overlap=0.01 # Initial overlap of the droplets
        self.delta=0.00001 # Initial transition of the c field
        self.remeshing_opts=RemeshingOptions(max_expansion=2,min_expansion=0.5)
        
        
    def define_problem(self):                        
        self+=SlightlyConnectedTwoDropletMesh()
        
        if self.Re==0:
            eqs=StokesEquations(dynamic_viscosity=1)
        else:
            eqs=NavierStokesEquations(mass_density=self.Re, dynamic_viscosity=1)        
            
        eqs+=AdvectionDiffusionEquations("c",diffusivity=1/self.Pe)+InitialCondition(c=0.5-atan(var("coordinate_x")/self.delta)/pi) 
        eqs+=HyperelasticSmoothedMesh() # Moving mesh equations
        eqs+=MeshFileOutput() # Output for Paraview (PVD [and VTU] files)

        # Boundary ocnditions        
        eqs+=( NavierStokesFreeSurface(surface_tension=1-self.beta*var("c")) + NavierStokesContactAngle(self.theta)@"substrate" )@"interface"
        eqs+=( DirichletBC(velocity_y=0,mesh_y=0) + NavierStokesSlipLength(self.slip) )@"substrate"
        
        # Remesh if required
        eqs+=RemeshWhen(self.remeshing_opts)
        
        self+=eqs@"liquid"
        
with NavierStokesCoalescenceProblem() as pr:    
    pr+=NeckLogger(bridge_half_width=0.25)
    pr+=DropletPlotter()
    pr.DTSF_max_increase_factor=1.1
    pr.run(200, outstep=True, startstep=0.0001, maxstep=0.1)
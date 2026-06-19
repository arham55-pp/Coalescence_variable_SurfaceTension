from pyoomph import *
from pyoomph.expressions import *

class LubricationEquations(Equations):
	"""
	Lubrication equations with insoluble surfactant transport.

	Implements dimensionless system (Eqs. 2.8-2.10 in paper):
	- Mass conservation for film height h
	- Pressure from curvature with surfactant-dependent surface tension
	- Surfactant transport by advection and diffusion

	Parameters:
		beta: Surfactant strength number (default 0.1)
		      β = a·Γ_∞/γ₀, maximal fractional reduction of surface tension
		Pe:   Surface Péclet number (default 1.0)
		      Pe = γ₀·L/(μ·D_s), advection/diffusion ratio
	"""
	def __init__(self, beta=0.1, Pe=1.0):
		super(LubricationEquations, self).__init__()
		self.beta = beta
		self.Pe = Pe
		
	def define_fields(self):
		self.define_scalar_field("h","C2")
		self.define_scalar_field("p","C2")		
		self.define_scalar_field("psi","C2") # 1D coordinate field
		
	def define_residuals(self):
		h,eta=var_and_test("h")
		p,q=var_and_test("p")
		psi,phi=var_and_test("psi")
		c=psi/h
		self.add_residual(weak(partial_t(h),eta) + weak(h**3/3*grad(p) + self.beta*h**2/2*grad(c), grad(eta)))
		self.add_residual(weak(p,q) - weak((1 - self.beta*c)*grad(h), grad(q)))
		self.add_residual(weak(partial_t(c*h),phi) + weak(h**3/3*c*grad(p) + self.beta*h**2/2*grad(c) + h*1/self.Pe*grad(c), grad(phi)))
"""		
class LubricationProblem(Problem):
	
	Simple test problem: thin film instability with surfactant.
	Not the coalescence simulation - just a sanity check for the equations.
	For calescence, use coalescence.py instead.
	
	def define_problem(self):
		self.add_mesh(LineMesh(N=100))
		eqs = LubricationEquations()  # uses defaults: beta=0.1, Pe=1.0
		eqs += TextFileOutput()
		# Small film with cosine perturbation to trigger instability
		eqs += InitialCondition(h=0.05 * (1 + 0.25 * cos(2 * pi * var("coordinate_x"))))
		eqs += InitialCondition(Gamma=0.5)  # uniform initial surfactant
		self.add_equations(eqs @ "domain")


if __name__ == "__main__":
	with LubricationProblem() as problem:
		problem.run(50, outstep=True, startstep=0.25)
"""
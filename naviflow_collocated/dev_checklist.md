SIMPLE Solver Developer Checklist (Moukalled-based)

📀 Numerical and Design Principles
	•	All variables (u, v, p) stored at cell centers
	•	Use Rhie–Chow interpolation on all internal faces
	•	Face-based assembly for all matrix coefficients (use Sf, d_cf, fx)
	•	Use Power Law for convection, central differencing for diffusion
	•	Follow Practice B for boundary conditions (modify matrix, not solution)
	•	Apply under-relaxation only after solving, not during matrix assembly
	•	Separate relaxed and unrelaxed field values for residual monitoring

⚙️ Core Architecture
	•	Each component (mesh, discretization, solver) lives in its own class
	•	All numerical choices (schemes, solvers) driven by YAML config
	•	Use abstract base classes for all interchangeable modules
	•	Avoid hardcoding numerical constants — use config or @dataclass

🥪 Testing & Debugging
	•	Write all tests to be mesh-agnostic (use mesh_instance fixture)
	•	Validate implementation on a 2x2 mesh before scaling up
	•	Use field min/max and residual logs for sanity checking
	•	Confirm diagonal dominance and matrix symmetry as applicable
	•	Include regression tests for Ghia benchmark (e.g. u_center ≈ 0.1565)

📄 Postprocessing & Logging
	•	Export intermediate fields (u, v, p) to .vtk or .csv
	•	Log residual history and solver convergence metrics
	•	Use INFO/DEBUG levels for selective verbosity

🧱 Extensibility
	•	New schemes and solvers must subclass the appropriate ABC
	•	Register new components in factory maps (e.g., SCHEME_REGISTRY)
	•	Design every loop to be parallel-ready: no global assumptions
	•	Avoid coupling logic across modules (e.g., discretization shouldn’t call PETSc)
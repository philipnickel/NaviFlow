from naviflow_collocated.mesh.structured_uniform import generate as gen_uniform
from naviflow_collocated.mesh.structured_refined import generate as gen_refined
from naviflow_collocated.mesh.unstructured import generate as gen_unstructured

# Generate meshes
uniform_mesh = gen_uniform(L=1.0, nx=50, ny=50)
refined_mesh = gen_refined(L=1.0, refine_edge='left', ratio=1.2)
unstructured_mesh = gen_unstructured(L=1.0, obstacle_radius=0.1)

# Save/load functionality would need to be added separately

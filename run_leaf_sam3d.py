"""Run SAM 3D on the diseased leaf image."""
import sys
import os

os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
sys.path.insert(0, "/work/mech-ai-scratch/UshB/sage-extend/sam-3d-objects/notebook")

from inference import Inference, load_image, load_single_mask

out_dir = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output"
os.makedirs(out_dir, exist_ok=True)

config_path = "/work/mech-ai-scratch/UshB/sage-extend/sam-3d-objects/checkpoints/hf/pipeline.yaml"
input_dir  = "/work/mech-ai-scratch/UshB/sage-extend/leaf_input"

print("Loading SAM 3D pipeline...")
inference = Inference(config_path, compile=False)
print("Pipeline loaded.")

image = load_image(os.path.join(input_dir, "image.png"))
mask  = load_single_mask(input_dir, index=0)
print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Mask coverage: {mask.mean()*100:.1f}%")

print("Running inference...")
output = inference(image, mask, seed=42)
print("Inference complete.")

ply_path = os.path.join(out_dir, "leaf_splat.ply")
output["gs"].save_ply(ply_path)
print(f"Gaussian splat saved to: {ply_path}")

# Also save mesh if present
if "mesh" in output:
    import trimesh
    meshes = output["mesh"]
    if not isinstance(meshes, list):
        meshes = [meshes]
    for i, m in enumerate(meshes):
        verts = m.vertices.cpu().numpy()
        faces = m.faces.cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        suffix = f"_{i}" if len(meshes) > 1 else ""
        mesh_path = os.path.join(out_dir, f"leaf_mesh{suffix}.obj")
        tm.export(mesh_path)
        print(f"Mesh saved to: {mesh_path}")

print("Done.")

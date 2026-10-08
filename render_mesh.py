"""Render leaf_mesh.obj to a PNG using matplotlib (no GPU/display needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import numpy as np

mesh_path = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_mesh.obj"
out_path  = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_mesh.png"

tm = trimesh.load(mesh_path, force="mesh")
tm.vertices -= tm.center_mass
tm.vertices /= max(tm.extents)

verts = tm.vertices
faces = tm.faces

# Compute face normals for shading
face_verts = verts[faces]
normals = np.cross(face_verts[:,1] - face_verts[:,0],
                   face_verts[:,2] - face_verts[:,0])
norms = np.linalg.norm(normals, axis=1, keepdims=True)
normals = normals / np.where(norms == 0, 1, norms)

light_dir = np.array([0.5, 0.8, 1.0])
light_dir /= np.linalg.norm(light_dir)
diffuse = np.clip(normals @ light_dir, 0, 1)
brightness = 0.25 + 0.65 * diffuse  # ambient + diffuse

colors = np.stack([brightness * 0.55, brightness * 0.75, brightness * 0.45], axis=1)
colors = np.clip(colors, 0, 1)

fig = plt.figure(figsize=(8, 8), facecolor="#1a1a1a")
ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

poly = Poly3DCollection(face_verts, facecolors=colors, linewidths=0, edgecolors="none")
ax.add_collection3d(poly)

lim = 0.6
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_axis_off()
ax.view_init(elev=25, azim=-60)

plt.tight_layout(pad=0)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")

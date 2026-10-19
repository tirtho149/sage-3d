"""Stage-1 arm / view-set constants shared by the generator (sam3d env) and the training code (Research .venv)."""
VIEWSETS = {"V8": [(0, 0), (15, 0), (-15, 0), (0, 15), (0, -15), (30, 0), (-30, 0), (0, 25)],
            "NARROW": [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (8, 4), (-8, -4), (4, -8)],
            "WIDE": [(0, 0), (45, 0), (-45, 0), (0, 40), (0, -40), (90, 0), (-90, 0), (180, 0)]}
ARMS3D = {"U3D": (None, False), "TEX3D": ("photo", False), "GEN3D": ("generic", False), "KBAPP3D": ("kb", False),
          "GEO3D": (None, True), "KB3D": ("kb", True), "TEXGEO3D": ("photo", True), "HARM3D": ("harm", False),
          "XKB3D": ("kbx", True), "XKBAPP3D": ("kbx", False)}
ARMS2D = {"PHOTO2D": "photo", "GEN2D": "generic", "KB2D": "kb", "PLATE2D": "plate",
          "XPHOTO2D": "photo", "XGEN2D": "generic", "XKB2D": "kbx"}
# X* = cross-class (knowledge-only) arms: the SIBLING class's parent (same crop, same organ), labelled with the target
# class; XKB* apply the target class's KB prompt (kbx edit) and KB geometry operators. XPHOTO2D / XGEN2D are controls.
CROSS = {"XPHOTO2D", "XGEN2D", "XKB2D", "XKB3D", "XKBAPP3D"}


def vkey(y, p):
    return f"y{y:+d}p{p:+d}"

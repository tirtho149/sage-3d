from PIL import Image
import numpy as np
img = Image.open('/work/mech-ai-scratch/UshB/sage-extend/Screenshot 2026-10-04 at 11.09.07 PM.png')
print('size:', img.size, 'mode:', img.mode)
arr = np.array(img)
print('shape:', arr.shape, 'dtype:', arr.dtype)
print('min/max R:', arr[...,0].min(), arr[...,0].max())
print('corner pixel (0,0):', arr[0,0])
print('center pixel:', arr[arr.shape[0]//2, arr.shape[1]//2])

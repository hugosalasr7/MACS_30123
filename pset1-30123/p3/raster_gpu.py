# Import required libraries
import rasterio
import numpy as np
import time
import pyopencl as cl
import pyopencl.array as cl_array

start = time.time()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Import bands as separate images; in /project2/macs30123 on Midway2
band4 = rasterio.open('/project2/macs30123/landsat8/LC08_B4.tif') #red
band5 = rasterio.open('/project2/macs30123/landsat8/LC08_B5.tif') #nir
# Convert nir and red objects to float64 arrays
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')
# Send them to GPU
red_dev = cl_array.to_device(queue, red)
nir_dev = cl_array.to_device(queue, nir)

# NDVI calculation
ndvi_dev = (nir_dev - red_dev) / (nir_dev + red_dev)
ndvi = ndvi_dev.get()

print(ndvi)
end = time.time()
print("GPU: Elapsed seconds = %s" % (end - start))
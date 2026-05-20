import numpy as np
import time
import gc

print("Initiating RAM sweep...")
print("Forcing background apps into NVMe swap by allocating 6GB of dummy data...")

dummy_data = []

# Allocate 6 chunks of 1GB each to prevent memory fragmentation crashes
for i in range(16):
    print(f"Allocating chunk {i+1}/6 (1GB)...")
    # 1024 * 1024 * 1024 bytes = 1GB of 8-bit integers
    dummy_data.append(np.ones((1024, 1024, 1024), dtype=np.uint8))
    # Give the kernel half a second to begin swapping other apps
    time.sleep(0.5)

print("6GB fully allocated. Holding for 3 seconds to ensure OS finishes swapping...")
time.sleep(3)

print("Freeing the allocated RAM...")
# Delete the reference to the massive arrays
del dummy_data
# Force Python's garbage collector to release the memory back to the OS instantly
gc.collect()

print("RAM cleared. Physical memory is now open for model loading.")

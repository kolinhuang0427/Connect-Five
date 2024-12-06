import ray
import time

# Connect to the Ray cluster
ray.init(address='auto')

@ray.remote
def heavy_computation(x):
    time.sleep(2)  # Simulate a heavy task
    return x * x

# Submit tasks to the cluster
results = ray.get([heavy_computation.remote(i) for i in range(10)])

print("Results:", results)

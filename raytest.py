import ray
import random
import time

# Initialize Ray
ray.init(address="auto")

@ray.remote
class TaskProcessor:
    def __init__(self):
        self.queue = []
        self.results = []

    def add_task(self, task):
        """Adds a task to the queue."""
        self.queue.append(task)
        print(f"Task '{task}' added to the queue.")

    def process_task(self):
        """Simulates task processing with a random delay."""
        if self.queue:
            task = self.queue.pop(0)
            processing_time = random.uniform(0.5, 2.0)  # Random processing time
            print(f"Processing task: '{task}'...")
            time.sleep(processing_time)
            result = f"Processed {task} in {processing_time:.2f}s"
            self.results.append(result)
            return result
        else:
            return "No tasks to process."

    def get_results(self):
        """Returns the results of processed tasks."""
        return self.results

# Create a TaskProcessor actor
processor = TaskProcessor.remote()

# Add tasks to the actor's queue
tasks = ["task_1", "task_2", "task_3", "task_4", "task_5"]
for task in tasks:
    processor.add_task.remote(task)

# Process tasks asynchronously
results = [processor.process_task.remote() for _ in tasks]

# Collect all results
processed_results = ray.get(results)

# Print processed results
print("Processed Results:")
for result in processed_results:
    print(result)

# Retrieve and print the internal results from the actor
final_results = ray.get(processor.get_results.remote())
print("\nFinal Results from Actor's Internal State:")
for result in final_results:
    print(result)

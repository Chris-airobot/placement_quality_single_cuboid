import json
import os
# Example JSON string
file_path = "/home/chris/Chris/placement_ws/src/random_data/Grasping_115/Placement_27_False.json"

# Parse the JSON string into a Python dictionary

with open(file_path, "r") as file:
    raw_data = json.load(file)  # Parse JSON into a Python dictionary



# Pretty-print the JSON
pretty_json = json.dumps(raw_data, indent=4)

directory, base_name = os.path.split(file_path)
pretty_name = "pretty_" + base_name
output_path = os.path.join("/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models", pretty_name)

with open(output_path, "w") as pretty_file:
    pretty_file.write(pretty_json)

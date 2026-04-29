import json
import os
from collections import defaultdict


def get_object_name(obj_string):
    """Extracts the object name from the string (part before the first '|')."""
    return obj_string.split("|")[0]


def save_results_as_json(results, scene_type):
    """Saves the results as a JSON file in the specified directory."""
    output_dir = os.path.join(
        "assets", "scene_knowledge", scene_type, "environment", "summary"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"common_objects_{scene_type}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def process_environment_files(directory_path, scene_type):
    """Reads JSON files, aggregates object counts by interaction type, and prints them."""
    # Structure: {interaction_type: {object_name: {'counts': [count_in_file1, ...], 'scenes': set(file_names)}}}
    interaction_counts = defaultdict(
        lambda: defaultdict(lambda: {"counts": [], "scenes": set()})
    )
    total_scenes = 0

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            total_scenes += 1
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Count objects within this specific file by interaction type
                for interaction_type, object_list in data.items():
                    for obj_string in object_list:
                        object_name = get_object_name(obj_string)
                        interaction_counts[interaction_type][object_name][
                            "counts"
                        ].append(1)
                        interaction_counts[interaction_type][object_name]["scenes"].add(
                            filename
                        )

            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from file '{filename}'. Skipping."
                )
            except Exception as e:
                print(
                    f"Warning: An error occurred while processing file '{filename}': {e}. Skipping."
                )

    # Prepare results for JSON output
    results = {"total_scenes": total_scenes, "interactions": {}}

    # Process each interaction type
    for interaction_type, objects in sorted(interaction_counts.items()):
        interaction_data = {"objects": []}

        # Only include objects that appear in all scenes
        for object_name, data in sorted(objects.items()):
            if len(data["scenes"]) == total_scenes:
                interaction_data["objects"].append(object_name)

        if interaction_data[
            "objects"
        ]:  # Only include interaction types that have common objects
            results["interactions"][interaction_type] = interaction_data

    # Print to console
    print(f"Total number of scenes: {total_scenes}")
    print("--- Objects that appear in all scenes by Interaction Type ---")
    for interaction_type, data in sorted(results["interactions"].items()):
        print(f"\nInteraction: {interaction_type}")
        print("Objects:")
        for object_name in sorted(data["objects"]):
            print(f"- {object_name}")
        print(f"Total objects: {len(data['objects'])}")

    print("\n----------------------------------------------------------")

    # Save results to JSON
    save_results_as_json(results, scene_type)


def find_non_common_objects(scene_type):
    """Finds non-common objects for each scene by action type."""
    # Read common objects file
    common_objects_file = os.path.join(
        "assets",
        "knowledge",
        scene_type,
        "environment",
        "summary",
        f"common_objects_{scene_type}.json",
    )
    with open(common_objects_file, "r") as f:
        common_objects = json.load(f)

    # Read concatenated scenes file
    concatenated_file = os.path.join(
        "assets",
        "knowledge",
        scene_type,
        "environment",
        "summary",
        f"concatenated_scenes_{scene_type}.json",
    )
    with open(concatenated_file, "r") as f:
        concatenated_scenes = json.load(f)

    # Prepare results structure
    results = {
        "scenes": {},
        "object_scene_appearances": {},  # Track how many scenes each object appears in and which scenes
    }

    # For each scene in concatenated scenes
    for scene_name, scene_data in concatenated_scenes.items():
        scene_results = {"actions": {}}

        # Track objects in this scene
        scene_objects = set()

        # For each action type in the scene
        for action_type, objects in scene_data.items():
            if action_type not in common_objects["interactions"]:
                # If action type doesn't exist in common objects, all objects are non-common
                # Count occurrences of each object
                object_counts = {}
                for obj in objects:
                    obj_name = get_object_name(obj)
                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                    scene_objects.add(obj_name)
                scene_results["actions"][action_type] = object_counts
            else:
                # Get common objects for this action type
                common_objs = set(
                    common_objects["interactions"][action_type]["objects"]
                )
                # Get objects in this scene for this action type
                scene_objs = {}
                for obj in objects:
                    obj_name = get_object_name(obj)
                    if obj_name not in common_objs:
                        scene_objs[obj_name] = scene_objs.get(obj_name, 0) + 1
                        scene_objects.add(obj_name)
                if scene_objs:
                    scene_results["actions"][action_type] = scene_objs

        if scene_results["actions"]:
            results["scenes"][scene_name] = scene_results

            # Update scene appearances for each object in this scene
            for obj_name in scene_objects:
                if obj_name not in results["object_scene_appearances"]:
                    results["object_scene_appearances"][obj_name] = {
                        "count": 0,
                        "scenes": set(),
                    }
                results["object_scene_appearances"][obj_name]["count"] += 1
                results["object_scene_appearances"][obj_name]["scenes"].add(scene_name)

    # Save non-common objects results
    output_file = os.path.join(
        "assets",
        "knowledge",
        scene_type,
        "environment",
        "summary",
        f"non_common_objects_{scene_type}.json",
    )
    with open(output_file, "w") as f:
        json.dump({"scenes": results["scenes"]}, f, indent=2)

    print(f"\nNon-common objects saved to: {output_file}")

    # Save object scene appearances to separate file
    appearances_file = os.path.join(
        "assets",
        "knowledge",
        scene_type,
        "environment",
        "summary",
        f"object_scene_appearances_{scene_type}.json",
    )
    serializable_appearances = {
        obj: {"count": data["count"], "scenes": sorted(list(data["scenes"]))}
        for obj, data in results["object_scene_appearances"].items()
    }
    with open(appearances_file, "w") as f:
        json.dump(serializable_appearances, f, indent=2)

    print(f"Object scene appearances saved to: {appearances_file}")

    # Print results to console
    print("\nNon-common objects by scene and action type:")
    for scene_name, scene_data in results["scenes"].items():
        print(f"\nScene: {scene_name}")
        for action_type, objects in scene_data["actions"].items():
            print(f"Action: {action_type}")
            total_count = 0
            for obj, count in sorted(objects.items()):
                print(f"- {obj}: {count} occurrences")
                total_count += count
            print(f"Total non-common objects: {total_count}")

    # Print object scene appearances
    print("\nObject scene appearances:")
    print("Object: number of scenes it appears in {scene numbers}")
    for obj, data in sorted(results["object_scene_appearances"].items()):
        scene_numbers = sorted(
            [int(s.replace("FloorPlan", "")) for s in data["scenes"]]
        )
        print(f"{obj}: {data['count']} scenes {{{', '.join(map(str, scene_numbers))}}}")


if __name__ == "__main__":
    scene_type = "kitchen"  # You can change this to other scene types
    knowledge_dir = f"assets/scene_knowledge/{scene_type}/environment"
    process_environment_files(knowledge_dir, scene_type)
    find_non_common_objects(scene_type)

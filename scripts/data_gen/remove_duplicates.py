import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def remove_duplicates():
    # Load the original data
    data = load_json('scripts/instructions_len_5.json')
    instructions = data['instructions']
    
    # Define kitchen and bathroom scenes
    kitchen_scenes = ['Floorplan1', 'Floorplan7', 'Floorplan13', 'Floorplan18', 'Floorplan27']
    bathroom_scenes = ['Floorplan401', 'Floorplan419', 'Floorplan422', 'Floorplan426', 'Floorplan427']
    
    # Get base instructions for kitchen and bathroom
    kitchen_base = set()
    bathroom_base = set()
    
    # Add all instructions from kitchen base
    for difficulty in ['simple', 'normal', 'complicated']:
        kitchen_base.update(instructions['Floorplan_kitchen'][difficulty])
    
    # Add all instructions from bathroom base
    for difficulty in ['simple', 'normal', 'complicated']:
        bathroom_base.update(instructions['Floorplan_bathroom'][difficulty])
    
    # Remove duplicates from kitchen scenes
    for scene in kitchen_scenes:
        if scene in instructions:
            for difficulty in ['simple', 'normal', 'complicated']:
                if difficulty in instructions[scene]:
                    # Remove instructions that exist in kitchen base
                    instructions[scene][difficulty] = [
                        instr for instr in instructions[scene][difficulty]
                        if instr not in kitchen_base
                    ]
    
    # Remove duplicates from bathroom scenes
    for scene in bathroom_scenes:
        if scene in instructions:
            for difficulty in ['simple', 'normal', 'complicated']:
                if difficulty in instructions[scene]:
                    # Remove instructions that exist in bathroom base
                    instructions[scene][difficulty] = [
                        instr for instr in instructions[scene][difficulty]
                        if instr not in bathroom_base
                    ]
    
    # Save the modified data
    save_json(data, 'scripts/instructions_len_5_no_duplicates.json')

if __name__ == "__main__":
    remove_duplicates() 
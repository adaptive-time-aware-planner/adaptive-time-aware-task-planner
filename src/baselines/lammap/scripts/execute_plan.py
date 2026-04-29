import os
from pathlib import Path
import subprocess
import argparse

def append_trans_ctr(allocated_plan):
    brk_ctr = 0
    code_segs = allocated_plan.split("\n\n")
    fn_calls = []
    for cd in code_segs:
        if "def" not in cd and "threading.Thread" not in cd and "join" not in cd and cd[-1] == ")":
            # fn_calls.append(cd)
            brk_ctr += 1
    print ("No Breaks: ", brk_ctr)
    return brk_ctr

def compile_aithor_exec_file(expt_name):
    log_path = os.getcwd() + "/logs/" + expt_name
    executable_plan = ""
    
    # append the imports to the file
    import_file = Path(os.getcwd() + "/data/aithor_connect/imports_aux_fn.py").read_text()
    executable_plan += (import_file + "\n")
    
    # append the list of robots and floor plan number
    log_file = open(log_path + "/log.txt")
    log_data = log_file.readlines()
    
    # Find robots line (contains "robots =")
    robots_line = None
    for line in log_data:
        if "robots = " in line:
            robots_line = line.strip()
            break
    
    if robots_line:
        executable_plan += (robots_line + "\n")
    else:
        # Fallback: create default robots list
        executable_plan += ("robots = [{'name': 'robot1', 'skills': ['GoToObject', 'PickupObject', 'PutObject'], 'mass': 100}]\n")
    
    # Set default floor number (since it's not in current log format)
    #! 무조건 floor 1로 설정되는데?
    executable_plan += ("floor_no = 209\n\n")
    
    # Find ground truth line (contains "ground_truth =")
    gt_line = None
    #! for line in log_data:
    #!     if "ground_truth = " in line:
    #!         gt_line = line.strip()
    #!         break
    
    #! if gt_line:
    #!     executable_plan += (gt_line + "\n")
    #! else:
    #!     # Fallback: create empty ground truth
    #! baek
    executable_plan += ("ground_truth = []\n")
    
    #! gt도 무조건 기본설정인데?
    # Set default trans values (since they're not in current log format)
    executable_plan += ("no_trans_gt = 0\n")
    executable_plan += ("max_trans = 10\n")
    
    # append the ai thoe connector and helper fns
    connector_file = Path(os.getcwd() + "/data/aithor_connect/aithor_connect.py").read_text()
    executable_plan += (connector_file + "\n")
    
    # append the allocated plan
    allocated_plan = Path(log_path + "/code_plan.py").read_text()
    
    # Replace empty robots list with actual robots from log file
    if robots_line:
        # Remove the robots = [] line from the generated code and replace with actual robots
        allocated_plan = allocated_plan.replace("robots = []", robots_line.strip())
        allocated_plan = allocated_plan.replace("robots = ['robot1']", robots_line.strip())
        allocated_plan = allocated_plan.replace("robots = ['Robot2']", robots_line.strip())
    
    brks = append_trans_ctr(allocated_plan)
    executable_plan += (allocated_plan + "\n")
    executable_plan += ("no_trans = " + str(brks) + "\n")

    # append the task thread termination
    terminate_plan = Path(os.getcwd() + "/data/aithor_connect/end_thread.py").read_text()
    executable_plan += (terminate_plan + "\n")

    with open(f"{log_path}/executable_plan.py", 'w') as d:
        d.write(executable_plan)
        
    return (f"{log_path}/executable_plan.py")

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, required=True)
args = parser.parse_args()

expt_name = args.command
print (expt_name)
ai_exec_file = compile_aithor_exec_file(expt_name)

subprocess.run(["python", ai_exec_file])
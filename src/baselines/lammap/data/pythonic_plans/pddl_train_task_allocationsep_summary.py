# EXAMPLE 1 - Task Description: Turn off the light, turn on the faucet and leave the house.
#GENERAL TASK DECOMPOSITION
#Decompose into sequential subtasks (single robot executes all subtasks one by one)
#Subtasks:
#SubTask 1: Turn off the Light. (Skills Required: GoToObject, ToggleObjectOff)
#SubTask 2: Turn on the Faucet. (Skills Required: GoToObject, ToggleObjectOn)
#SubTask 3: Leave the House. (Skills Required: GoToObject, OpenObject, CloseObject)
#SubTask 3 is dependent on SubTask 1 and 2 being completed. All subtasks executed sequentially by the single robot.

#action description from domain for tasks required
#Subtask 1: Turn off the Light

# Initial Precondition analyze due to previous subtask:
#1. Robot not at light switch location

GoToObject: Robot goes to the light switch.
Parameters: ?robot, ?lightSwitch
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?lightSwitch), (not (inaction ?robot))

ToggleObjectOff: Robot turns off the light.
Parameters: ?robot, ?lightSwitch
Preconditions: (at ?robot ?lightSwitch), (not (inaction ?robot))
Effects: (toggled-off ?robot ?lightSwitch), (not (inaction ?robot))

#Subtask 2: Turn on the Faucet

# Initial Precondition analyze due to previous subtask:
#1. Robot not at Faucet location

GoToObject: Robot goes to the faucet.
Parameters: ?robot, ?faucet
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?faucet), (not (inaction ?robot))

ToggleObjectOn: Robot turns on the faucet.
Parameters: ?robot, ?faucet
Preconditions: (at ?robot ?faucet), (not (inaction ?robot))
Effects: (toggled-on ?robot ?faucet), (not (inaction ?robot))

#Subtask 3: Leave the House

# Initial Precondition analyze due to previous subtask:
#1. Robot not at the door
GoToObject: Robot goes to the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?door), (not (inaction ?robot))

OpenObject: Robot opens the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot)), (at ?robot ?door)
Effects: (object-open ?robot ?door), (not (inaction ?robot))

CloseObject: Robot closes the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot)), (at ?robot ?door)
Effects: (object-close ?robot ?door), (not (inaction ?robot))

# TASK ALLOCATION
# Scenario: There is 1 robot available with all skills. All subtasks are assigned to Robot 1 and executed sequentially.
robots = [{'name': 'robot1', 'skills': ['GoToObject', 'PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'CleanObject'], 'mass': 100}]
objects = [{'name': 'SaltShaker', 'mass': 1.0}, {'name': 'SoapBottle', 'mass': 5.0}, {'name': 'Bowl', 'mass': 3.0}, {'name': 'CounterTop', 'mass': 2}, {'name': 'Drawer', 'mass': 5}, {'name': 'Vase', 'mass': 1}, {'name': 'Lettuce', 'mass': 5}, {'name': 'Plate', 'mass': 5}, {'name': 'Potato', 'mass': 2}, {'name': 'Pan', 'mass': 5}, {'name': 'Window', 'mass': 4}, {'name': 'Fridge', 'mass': 4}, {'name': 'Sink', 'mass': 5}, {'name': 'Stool', 'mass': 6}, {'name': 'DiningTable', 'mass': 6}, {'name': 'HousePlant', 'mass': 2}, {'name': 'Chair', 'mass': 6}, {'name': 'Cup', 'mass': 3}, {'name': 'Shelf', 'mass': 6}, {'name': 'StoveKnob', 'mass': 3}, {'name': 'ButterKnife', 'mass': 6}, {'name': 'Bread', 'mass': 2}, {'name': 'DishSponge', 'mass': 4}, {'name': 'Floor', 'mass': 4}, {'name': 'PepperShaker', 'mass': 2}, {'name': 'Spatula', 'mass': 2}, {'name': 'StoveBurner', 'mass': 4}, {'name': 'Statue', 'mass': 1}, {'name': 'Kettle', 'mass': 2}, {'name': 'Apple', 'mass': 4}, {'name': 'WineBottle', 'mass': 5}, {'name': 'Knife', 'mass': 6}, {'name': 'GarbageCan', 'mass': 5}, {'name': 'Microwave', 'mass': 5}, {'name': 'LightSwitch', 'mass': 6}, {'name': 'Pot', 'mass': 2}, {'name': 'Egg', 'mass': 3}, {'name': 'Mug', 'mass': 2}, {'name': 'CoffeeMachine', 'mass': 6}, {'name': 'Faucet', 'mass': 6}, {'name': 'Tomato', 'mass': 4}, {'name': 'Spoon', 'mass': 2}, {'name': 'Fork', 'mass': 4.8}, {'name': 'Toaster', 'mass': 1}, {'name': 'Book', 'mass': 6}, {'name': 'SinkBasin', 'mass': 6}, {'name': 'Cabinet', 'mass': 1}, {'name': 'ShelvingUnit', 'mass': 3}]

# SOLUTION
# There is only 1 robot (Robot 1) with all skills. All subtasks are assigned to Robot 1 sequentially.

#Problem content summary (generate strictly in example format)

#SubTask 1: Turn off the Light

# Initial Precondition analyze due to previous subtask:
#1. Robot not at light switch location
GoToObject: Robot goes to the light switch.
Parameters: ?robot, ?lightSwitch
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?lightSwitch), (not (inaction ?robot))

ToggleObjectOff: Robot turns off the light.
Parameters: ?robot, ?lightSwitch
Preconditions: (at ?robot ?lightSwitch), (not (inaction ?robot))
Effects: (toggled-off ?robot ?lightSwitch), (not (inaction ?robot))


Assigned Robot: Robot 1
Objects Involved: LightSwitch


#SubTask 2: Turn on the Faucet (generate strictly in example format)

# Initial Precondition analyze due to previous subtask:
#1. Robot not at Faucet location
GoToObject: Robot goes to the faucet.
Parameters: ?robot, ?faucet
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?faucet), (not (inaction ?robot))

ToggleObjectOn: Robot turns on the faucet.
Parameters: ?robot, ?faucet
Preconditions: (at ?robot ?faucet), (not (inaction ?robot))
Effects: (toggled-on ?robot ?faucet), (not (inaction ?robot))


Assigned Robot: Robot 1
Objects Involved: Faucet


#SubTask 3: Leave the House (generate strictly in example format)

# Initial Precondition analyze due to previous subtask:
#1. Robot not at the door

GoToObject: Robot goes to the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?door), (not (inaction ?robot))

OpenObject: Robot opens the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot)), (at ?robot ?door)
Effects: (object-open ?robot ?door), (not (inaction ?robot))

CloseObject: Robot closes the door.
Parameters: ?robot, ?door
Preconditions: (not (inaction ?robot)), (at ?robot ?door)
Effects: (object-close ?robot ?door), (not (inaction ?robot))


Assigned Robot: Robot 1
Objects Involved: Door


#Given the constraints and available resources, the task allocation can be broken down as follows:

#Sequence of Operations

    All subtasks are performed sequentially by Robot 1.
    SubTask 1 Turn off the Light is performed first.
    SubTask 2 Turn on the Faucet is performed second.
    SubTask 3 Leave the House is performed last after SubTask 1 and 2 are completed.


#Problem content summary done
#IMPORTANT strictly follow the structure and do not generate beyond Problem content summary done.



#Task Description: Slice the Potato
#GENERAL TASK DECOMPOSITION
#Decompose into sequential subtasks (single robot executes all subtasks one by one)
#SubTask: Slice the Potato (Skills Required: GoToObject, PickupObject, GoToObject, SliceObject)
#Subtask 1 is the only task. Executed sequentially by Robot 1.

#action description from domain for tasks required

#Subtask 1: Slice the Potato

# Initial Precondition analyze due to previous subtask:
#1. Robot not at the potato
#2. Robot not holding the potato

GoToObject (Robot, Knife)
Parameters: ?robot , ?Knife  ,
Preconditions: (not (inaction Robot))
Effects: (at Robot CuttingBoard), (not (inaction Robot))

PickupObject (Robot, Knife, KnifeLocation)
Parameters: ?robot , ?Knife , ?KnifeLocation
Preconditions: (at-location Knife KnifeLocation), (at Robot KnifeLocation), (not (inaction Robot))
Effects: (holding Robot Knife), (not (inaction Robot))

GoToObject: Robot goes to the potato.
Parameters: ?robot, ?potato
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?potato), (not (inaction ?robot))

SliceObject: Robot slices the potato.
Parameters: ?robot, ?potato
Preconditions: (holding Robot Knife), (not (inaction ?robot))
Effects: (sliced ?potato), (not (inaction ?robot))


Assigned Robot: Robot 1
Objects Involved: Potato, knife


#Sequence of Operations:

    SubTask Slice the Potato is performed by Robot 1.

#Problem content summary done
#IMPORTANT strictly follow the structure and do not generate beyond Problem content summary done.

# Single-agent configuration (allactionrobot.pddl 기반)

robot1 = {
    'name': 'robot1',
    'skills': [
        'GoToObject', 'PickupObject', 'PutObject',
        'ToggleObjectOn', 'ToggleObjectOff',
        'OpenObject', 'CloseObject', 'SliceObject',
        # 'BreakObject', 'CleanObject',
        'OpenFridge', 'CloseFridge',
    ],
    'mass': 100
}

robots = [robot1]



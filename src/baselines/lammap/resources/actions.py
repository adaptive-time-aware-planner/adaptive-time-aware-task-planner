ai2thor_actions = [
    "GoToObject <robot><object>",
    "PickupObject <robot><object><location>",
    "PutObject <robot><object><receptacleObject>",
    "ToggleObjectOn <robot><object>",
    "ToggleObjectOff <robot><object>",
    "OpenObject <robot><object>",
    "CloseObject <robot><object>",
    "SliceObject <robot><object><location>",
    # "CleanObject <robot><object>",
    # "BreakObject <robot><object>",
]
ai2thor_actions = ', '.join(ai2thor_actions)

def pick_up_with_arm(controller):
    """
    Picks up an object from the current position of the arm.
    """
    try:
        # Get list of objects that can be picked up from the current arm position
        pickupable_objects = controller.last_event.metadata["arm"].get(
            "pickupableObjects", []
        )
        if not pickupable_objects:
            print("No pickupable objects nearby.")
            return

        event = controller.step(
            action="PickupObject", objectIdCandidates=pickupable_objects
        )
        if event.metadata["lastActionSuccess"]:
            print(f"Picked up object: {pickupable_objects[0]}")
        else:
            print(event.metadata["errorMessage"])
    except Exception as e:
        print(f"Error during PickupObject action: {str(e)}")


def drop_object(controller):
    """
    Drops the currently held object.
    """
    try:
        event = controller.step(action="ReleaseObject")
        if event.metadata["lastActionSuccess"]:
            print("Object has been dropped.")
        else:
            print("Cannot drop the object.")
    except Exception as e:
        print(f"Error during ReleaseObject action: {str(e)}")


FORCE = False


def pick_up(controller, objectId, force=FORCE):
    controller.step(action="PickupObject", objectId=objectId, forceAction=force)


def put(controller, objectId, force=FORCE):
    controller.step(action="PutObject", objectId=objectId, forceAction=force)


def drop(controller, objectId, force=FORCE):
    controller.step(action="DropObject", objectId=objectId, forceAction=force)


def toggle_on(controller, objectId, force=FORCE):
    controller.step(action="ToggleObjectOn", objectId=objectId, forceAction=force)


def toggle_off(controller, objectId, force=FORCE):
    controller.step(action="ToggleObjectOff", objectId=objectId, forceAction=force)


def slice(controller, objectId, force=FORCE):
    controller.step(action="SliceObject", objectId=objectId, forceAction=force)


def open(controller, objectId, force=FORCE):
    controller.step(action="OpenObject", objectId=objectId, forceAction=True)


def close(controller, objectId, force=FORCE):
    controller.step(action="CloseObject", objectId=objectId, forceAction=True)


def teleport_to_position(controller, position):
    """
    Teleports the agent to the specified position.
    """
    controller.step(
        action="Teleport",
        position={"x": position[0], "y": position[1], "z": position[2]},
    )

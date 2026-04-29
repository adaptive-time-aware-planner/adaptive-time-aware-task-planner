from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ObjectCondition:
    """Object condition with name prefix matching and required properties."""

    object_name_prefix: str
    required_properties: Mapping[str, Any]


@dataclass(frozen=True)
class ConditionGroup:
    """A set of object conditions that must be true simultaneously in a step."""

    objects: Sequence[ObjectCondition]


@dataclass(frozen=True)
class TSRSpec:
    """Single TSR specification with trigger and end conditions."""

    name: str  # e.g., "fill", "boil", "heat"
    trigger: ConditionGroup
    end: ConditionGroup


@dataclass(frozen=True)
class TaskSpec:
    """Specification to evaluate a task's GCR/TSR.

    A task can have multiple TSRs (e.g., boil_potato has 'fill' and 'boil' TSRs).
    For backward compatibility, tsr_trigger/tsr_end are still supported.
    """

    name: str
    gcr_end: Optional[ConditionGroup] = None
    gcr_mid_groups: Optional[Sequence[ConditionGroup]] = None
    # Legacy single TSR support (for backward compatibility)
    tsr_trigger: Optional[ConditionGroup] = None
    tsr_end: Optional[ConditionGroup] = None
    # New multiple TSR support
    tsrs: Optional[Sequence[TSRSpec]] = None


def OC(name_prefix: str, **props: Any) -> ObjectCondition:
    """Helper to build ObjectCondition."""

    return ObjectCondition(object_name_prefix=name_prefix, required_properties=props)


def CG(*objs: ObjectCondition) -> ConditionGroup:
    """Helper to build ConditionGroup."""

    return ConditionGroup(objects=list(objs))


def TSR(name: str, trigger: ConditionGroup, end: ConditionGroup) -> TSRSpec:
    """Helper to build TSRSpec."""

    return TSRSpec(name=name, trigger=trigger, end=end)


TASK_SPECS: Dict[str, TaskSpec] = {
    # boil_potato (두 개의 TSR: fill + boil)
    "boil_potato": TaskSpec(
        name="boil_potato",
        gcr_end=CG(OC("potato", isCooked=True)),
        gcr_mid_groups=[
            CG(
                # OC("potato", parentReceptacles=["pot"]),
                OC("pot_", isFilledWithLiquid=True, parentReceptacles=["stove"]),
                OC("stoveknob", isToggled=True),
            )
        ],
        # Multiple TSRs: 1) fill water, 2) boil
        tsrs=[
            TSR(
                name="fill_water_in_pot",
                trigger=CG(
                    OC("pot_", isFilledWithLiquid=True),
                    OC("faucet", isToggled=True),
                ),
                end=CG(OC("faucet", isToggled=False)),
            ),
            TSR(
                name="boil_potato",
                trigger=CG(
                    OC("potato", isCooked=True),
                    OC("pot_", isFilledWithLiquid=True, parentReceptacles=["stove"]),
                    OC("stoveknob", isToggled=True),
                ),
                end=CG(OC("stoveknob", isToggled=False)),
            ),
        ],
    ),
    # boil_water_with_pot (두 개의 TSR: fill + boil)
    "boil_water_with_pot": TaskSpec(
        name="boil_water_with_pot",
        gcr_end=CG(OC("pot_", isFilledWithLiquid=True)),
        gcr_mid_groups=[
            CG(
                OC("pot_", isFilledWithLiquid=True, parentReceptacles=["stove"]),
                OC("stoveknob", isToggled=True),
            )
        ],
        # Multiple TSRs: 1) fill water, 2) boil
        tsrs=[
            TSR(
                name="fill_water_in_pot",
                trigger=CG(
                    OC("pot_", isFilledWithLiquid=True),
                    OC("faucet", isToggled=True),
                ),
                end=CG(OC("faucet", isToggled=False)),
            ),
            TSR(
                name="boil_water_in_pot",
                trigger=CG(
                    OC("pot_", isFilledWithLiquid=True, parentReceptacles=["stove"]),
                    OC("stoveknob", isToggled=True),
                ),
                end=CG(OC("stoveknob", isToggled=False)),
            ),
        ],
    ),
    # fill_pot_with_water
    "fill_pot_with_water": TaskSpec(
        name="fill_pot_with_water",
        gcr_end=CG(OC("pot_", isFilledWithLiquid=True)),
        gcr_mid_groups=[CG(OC("pot_", isFilledWithLiquid=True))],
        tsrs=[
            TSR(
                name="fill_water_in_pot",
                trigger=CG(
                    OC("pot_", isFilledWithLiquid=True),
                    OC("faucet", isToggled=True),
                ),
                end=CG(OC("faucet", isToggled=False)),
            ),
        ],
    ),
    # fill_bowl_with_water
    "fill_bowl_with_water": TaskSpec(
        name="fill_bowl_with_water",
        gcr_end=CG(OC("bowl", isFilledWithLiquid=True)),
        gcr_mid_groups=[CG(OC("bowl", isFilledWithLiquid=True))],
        tsrs=[
            TSR(
                name="fill_water_in_bowl",
                trigger=CG(
                    OC("bowl", isFilledWithLiquid=True),
                    OC("faucet", isToggled=True),
                ),
                end=CG(OC("faucet", isToggled=False)),
            ),
        ],
    ),
    # heat_the_bread_using_microwave
    "heat_the_bread_using_microwave": TaskSpec(
        name="heat_the_bread_using_microwave",
        gcr_end=None,
        gcr_mid_groups=[
            CG(
                OC("bread", parentReceptacles=["microwave"]),
                OC("microwave", isToggled=True),
            )
        ],
        tsrs=[
            TSR(
                name="heat_bread_in_microwave",
                trigger=CG(
                    OC("bread", parentReceptacles=["microwave"]),
                    OC("microwave", isToggled=True),
                ),
                end=CG(OC("microwave", isToggled=False)),
            ),
        ],
    ),
    # heat_the_potato_using_microwave
    "heat_the_potato_using_microwave": TaskSpec(
        name="heat_the_potato_using_microwave",
        gcr_end=None,
        gcr_mid_groups=[
            CG(
                OC("potato", parentReceptacles=["microwave"]),
                OC("microwave", isToggled=True),
            )
        ],
        tsrs=[
            TSR(
                name="heat_potato_in_microwave",
                trigger=CG(
                    OC("potato", parentReceptacles=["microwave"]),
                    OC("microwave", isToggled=True),
                ),
                end=CG(OC("microwave", isToggled=False)),
            ),
        ],
    ),
    # make_a_coffee
    "make_a_coffee": TaskSpec(
        name="make_a_coffee",
        gcr_end=CG(OC("mug", isFilledWithLiquid=True)),
        gcr_mid_groups=[
            CG(
                OC("mug", isFilledWithLiquid=True, parentReceptacles=["CoffeeMachine"]),
                OC("CoffeeMachine", isToggled=True),
            )
        ],
        tsrs=[
            TSR(
                name="make_coffee",
                trigger=CG(
                    OC(
                        "mug",
                        isFilledWithLiquid=True,
                        parentReceptacles=["CoffeeMachine"],
                    ),
                    OC("CoffeeMachine", isToggled=True),
                ),
                end=CG(
                    OC("mug", isFilledWithLiquid=True, parentReceptacles=None),
                ),
            ),
        ],
    ),
    # cook_egg
    "cook_egg": TaskSpec(
        name="cook_egg",
        gcr_end=CG(OC("Egg_Cracked", isCooked=True)),
        gcr_mid_groups=[
            CG(
                # OC("egg_sliced", parentReceptacles=["pan"]),
                OC("pan", parentReceptacles=["stove"]),
                OC("Egg_Cracked", isCooked=True),
                OC("stoveknob", isToggled=True),
            )
        ],
        tsrs=[
            TSR(
                name="heat_egg",
                trigger=CG(
                    # OC("egg_sliced", parentReceptacles=["pan"]),
                    OC("Egg_Cracked", isCooked=True),
                    OC("pan", parentReceptacles=["stove"]),
                    OC("stoveknob", isToggled=True),
                ),
                end=CG(OC("stoveknob", isToggled=False)),
            ),
        ],
    ),
    # put_apple_and_lettuce_in_fridge
    "put_apple_and_lettuce_in_fridge": TaskSpec(
        name="put_apple_and_lettuce_in_fridge",
        gcr_end=CG(
            OC("apple", parentReceptacles=["fridge"]),
            OC("lettuce", parentReceptacles=["fridge"]),
        ),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    # wash_a_tomato
    "wash_a_tomato": TaskSpec(
        name="wash_a_tomato",
        gcr_end=None,
        gcr_mid_groups=[
            CG(OC("tomato", parentReceptacles=["sink"]), OC("faucet", isToggled=True))
        ],
        tsrs=None,
    ),
    # wash_a_butterknife
    "wash_a_butterknife": TaskSpec(
        name="wash_a_butterknife",
        gcr_end=None,
        gcr_mid_groups=[
            CG(
                OC("butterknife", parentReceptacles=["sink"]),
                OC("faucet", isToggled=True),
            )
        ],
        tsrs=None,
    ),
    # wash_a_spatula
    "wash_a_spatula": TaskSpec(
        name="wash_a_spatula",
        gcr_end=None,
        gcr_mid_groups=[
            CG(OC("spatula", parentReceptacles=["sink"]), OC("faucet", isToggled=True))
        ],
        tsrs=None,
    ),
    # wash_all_fork_and_spoon (각각 독립 그룹 2개)
    "wash_all_fork_and_spoon": TaskSpec(
        name="wash_all_fork_and_spoon",
        gcr_end=None,
        gcr_mid_groups=[
            CG(OC("fork", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
            CG(OC("spoon", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
        ],
        tsrs=None,
    ),
    # wash_plate_and_cup (각각 독립 그룹 2개)
    "wash_plate_and_cup": TaskSpec(
        name="wash_plate_and_cup",
        gcr_end=None,
        gcr_mid_groups=[
            CG(OC("plate", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
            CG(OC("cup", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
        ],
        tsrs=None,
    ),
    # wash_apple_and_lettuce (각각 독립 그룹 2개)
    "wash_apple_and_lettuce": TaskSpec(
        name="wash_apple_and_lettuce",
        gcr_end=None,
        gcr_mid_groups=[
            CG(OC("apple", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
            CG(OC("lettuce", parentReceptacles=["sink"]), OC("faucet", isToggled=True)),
        ],
        tsrs=None,
    ),
    # cook_chicken
    "cook_chicken": TaskSpec(
        name="cook_chicken",
        gcr_end=None,
        gcr_mid_groups=[
            CG(
                OC("chicken", parentReceptacles=["pan"]),
                OC("pan", parentReceptacles=["stove"]),
                OC("stove", isToggled=True),
            ),
        ],
        tsrs=[
            TSR(
                name="cook_chicken",
                trigger=CG(
                    OC("chicken", parentReceptacles=["pan"]),
                    OC("pan", parentReceptacles=["stove"]),
                    OC("stove", isToggled=True),
                ),
                end=CG(OC("stove", isToggled=False)),
            ),
        ],
    ),
    # do_the_laundry
    "do_the_laundry": TaskSpec(
        name="do_the_laundry",
        gcr_end=CG(OC("laundry_machine", isToggled=False)),
        gcr_mid_groups=[CG(OC("laundry_machine", isToggled=True))],
        tsrs=[
            TSR(
                name="do_laundry",
                trigger=CG(OC("laundry_machine", isToggled=True)),
                end=CG(OC("laundry_machine", isToggled=False)),
            ),
        ],
    ),
    # cook_sausage
    "cook_sausage": TaskSpec(
        name="cook_sausage",
        gcr_end=None,
        gcr_mid_groups=[
            CG(
                OC("sausage", parentReceptacles=["pan"]),
                OC("pan", parentReceptacles=["stove"]),
                OC("stove", isToggled=True),
            ),
        ],
        tsrs=[
            TSR(
                name="cook_sausage",
                trigger=CG(
                    OC("sausage", parentReceptacles=["pan"]),
                    OC("pan", parentReceptacles=["stove"]),
                    OC("stove", isToggled=True),
                ),
                end=CG(OC("stove", isToggled=False)),
            ),
        ],
    ),
    "move_cup_to_sink": TaskSpec(
        name="move_cup_to_sink",
        gcr_end=CG(OC("cup", parentReceptacles=["sink"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    "place_pan_on_stove": TaskSpec(
        name="place_pan_on_stove",
        gcr_end=CG(OC("pan", parentReceptacles=["stove"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    "put_banana_on_plate": TaskSpec(
        name="put_banana_on_plate",
        gcr_end=CG(OC("banana", parentReceptacles=["plate"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    "put_orange_bowl_on_sink": TaskSpec(
        name="put_orange_bowl_on_sink",
        gcr_end=CG(OC("orange_bowl", parentReceptacles=["sink"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    "move_blue_bowl_to_sink": TaskSpec(
        name="move_blue_bowl_to_sink",
        gcr_end=CG(OC("blue_bowl", parentReceptacles=["sink"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
    "put_tomato_on_plate": TaskSpec(
        name="put_tomato_on_plate",
        gcr_end=CG(OC("tomato", parentReceptacles=["plate"])),
        gcr_mid_groups=None,
        tsrs=None,
    ),
}

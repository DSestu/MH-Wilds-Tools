from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Dict, List

import polars as pl
from icecream import ic
from ortools.sat.python import cp_model
from rich import traceback

traceback.install()

armor = pl.read_parquet("data/armor_pieces.parquet")
charms = pl.read_parquet("data/charms.parquet")
jewels = pl.read_parquet("data/jewels.parquet")
talents = pl.read_parquet("data/talents.parquet")
weapons = pl.read_parquet("data/weapons.parquet")


@dataclass
class OptimizationVariables:
    # Stores boolean variables indicating whether each armor piece is used
    use_armor_piece_booleans: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Stores boolean variables indicating whether each charm is used
    use_charm_booleans: Dict[str, Any] = field(default_factory=dict)
    # Lists of talent levels for each talent
    talent_lists: Dict[str, List[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Sum of talent levels for each talent
    talent_sums: Dict[str, Any] = field(default_factory=dict)
    # Sum of talent levels capped at a maximum value for each talent
    talent_sums_capped: Dict[str, Any] = field(default_factory=dict)
    # Series of talent levels within specified intervals
    talent_series_interval: Dict[str, List[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Final sum of talent levels after all calculations
    talent_sums_final: Dict[str, Any] = field(default_factory=dict)
    # Boolean indicating if a group has enough talent level
    group_has_enough_level: Dict[str, Any] = field(default_factory=dict)
    # Lists of jewel emplacement options for each armor piece
    jewel_emplacement_lists: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    # Sum of jewel emplacement values for each armor piece
    jewel_emplacement_sums: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Total sum of jewel emplacement values for all armor pieces
    jewel_emplacement_sums_total_armor: Dict[str, Any] = field(default_factory=dict)
    # Total sum of jewel emplacement values for all weapons
    jewel_emplacement_sums_total_weapon: Dict[str, Any] = field(default_factory=dict)
    # Integer variables representing jewel usage for each level
    jewel_uses_integers: Dict[int, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )


def _process_armor_pieces(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    gear_type: str,
) -> None:
    """
    Processes each unique armor piece to set up constraint programming variables and constraints.

    This function iterates over each armor piece in the provided list of unique pieces. For each armor piece, it:
    - Initializes dictionaries in the _vars dictionary to store boolean variables and jewel emplacement data.
    - Filters the armor data to get only the rows corresponding to the current armor piece.
    - Retrieves unique armor piece names and iterates over them.
    - For each unique armor piece name, it creates a boolean variable indicating whether the armor piece is equipped.
    - Iterates over each row of the filtered armor piece data to:
        - Create an integer variable representing the talent level activated by equipping the armor piece.
        - Add constraints to enforce the talent level based on whether the armor piece is equipped.
        - Register the talent level variable in the _vars dictionary.
    - Stores the boolean variable indicating if the armor piece is equipped in the _vars dictionary.

    Args:
        model (cp_model.CpModel): The constraint programming model to which variables and constraints are added.
        _vars (dict): A dictionary to store variables and other data used in the constraint programming model.
        gear_type (str): The name of the type of armor piece being processed.

    Returns:
        None
    """
    # Filter the armor data to get only the rows corresponding to the current armor piece
    armor_piece_filtered = armor.filter(pl.col("piece") == gear_type)
    # Retrieve unique armor piece names and sort them into a list
    unique_armor_pieces_names = armor_piece_filtered["name"].unique().sort().to_list()
    for unique_armor_piece_name in unique_armor_pieces_names:
        # Filter to get data for the specific unique armor piece name
        unique_armor_piece = armor_piece_filtered.filter(
            pl.col("name") == unique_armor_piece_name
        )

        # Define a boolean variable that indicates if the armor piece is equipped
        names = unique_armor_piece["name"].to_list()
        # There are multiple rows for the same armor piece, because each row is a potential talent, so we take the first one to get the piece name, but they are all the same
        name = names[0]
        armor_piece_equipped = model.NewBoolVar(f"use_piece_{gear_type}_{name}")

        for row in unique_armor_piece.iter_rows():
            (name, talent_name, talent_level) = row[1:4]
            # Create an integer variable representing the talent level activated by equipping the armor piece
            var_talent_lvl = model.NewIntVar(
                lb=0,
                ub=30,
                name=f"talent_{talent_name}_from_type_{gear_type}_with_{name}",
            )
            # Add constraints to enforce the talent level based on whether the armor piece is equipped
            model.Add(var_talent_lvl == talent_level).only_enforce_if(
                armor_piece_equipped
            )
            # If the armor piece is not equipped, the talent level should be 0
            model.Add(var_talent_lvl == 0).only_enforce_if(armor_piece_equipped.Not())

            # Register the talent level variable in the _vars dictionary
            _vars.talent_lists[talent_name].append(var_talent_lvl)

        # Store the boolean variable indicating if the armor piece is equipped in the _vars dictionary
        _vars.use_armor_piece_booleans[gear_type][name] = armor_piece_equipped
    # Add the constraint of only one type of armor piece equipped at a time
    model.Add(sum(_vars.use_armor_piece_booleans[gear_type].values()) <= 1)


def _create_jewel_slots_for_armor_pieces(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    gear_type: str,
) -> None:
    """
    Creates jewel slots for a given armor piece within the optimization model.

    This function processes each unique armor piece name to handle jewel emplacements.
    It defines boolean variables indicating if the armor piece is equipped and creates
    integer variables for each jewel level. Constraints are added to enforce the jewel
    levels based on whether the armor piece is equipped.

    Args:
        model (cp_model.CpModel): The optimization model to which constraints are added.
        _vars (OptimizationVariables): A data structure holding variables used in the model.
        gear_type (str): The name of the type of armor piece being processed.

    Returns:
        None
    """
    # Filter the armor data to get only the rows corresponding to the current armor piece
    armor_piece_filtered = armor.filter(pl.col("piece") == gear_type)
    # Retrieve unique armor piece names and sort them into a list
    unique_armor_pieces_names = armor_piece_filtered["name"].unique().sort().to_list()

    # Iterate over each unique armor piece name to process jewel emplacements
    for unique_armor_piece_name in unique_armor_pieces_names:
        # Filter to get data for the specific unique armor piece name
        unique_armor_piece = armor_piece_filtered.filter(
            pl.col("name") == unique_armor_piece_name
        )
        names = unique_armor_piece["name"].to_list()
        # There are multiple rows for the same armor piece, because each row is a potential talent, so we take the first one to get the piece name, but they are all the same
        name = names[0]

        # Retrieve the boolean variable indicating if the armor piece is equipped
        armor_piece_equipped = _vars.use_armor_piece_booleans[gear_type][name]

        # Process each row in the unique armor piece data to handle jewel levels
        for row in unique_armor_piece.iter_rows():
            (_, name, talent_name, talent_level, *jewels) = row
            # Unpack jewel levels from the row data
            jewel_0, jewel_1, jewel_2, jewel_3, jewel_4 = jewels

            # Initialize jewel levels and corresponding variables
            jewel_levels = [jewel_1, jewel_2, jewel_3, jewel_4]
            var_nb_jewels = []

            # Create integer variables for each jewel level and add constraints
            for i, jewel in enumerate(jewel_levels, start=1):
                var_nb_jewel = model.NewIntVar(
                    lb=0,
                    ub=4,
                    name=f"jewel_lvl_{i}_from_type_{gear_type}_with_{name}",
                )
                # Enforce constraints based on whether the armor piece is equipped
                model.Add(var_nb_jewel == jewel).only_enforce_if(armor_piece_equipped)
                model.Add(var_nb_jewel == 0).only_enforce_if(armor_piece_equipped.Not())
                var_nb_jewels.append(var_nb_jewel)

            # Register the jewel level variables in the _vars dictionary
            for lvl, var_nb_jewel in zip(
                ["lvl1", "lvl2", "lvl3", "lvl4"],
                var_nb_jewels,
            ):
                _vars.jewel_emplacement_lists[gear_type][lvl].append(var_nb_jewel)

    # Create variables that are the sum of the jewel emplacements for each level
    for i in range(1, 5):
        lvl = f"lvl{i}"
        var_jewel_emplacement_sums = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"jewel_emplacement_sums_{lvl}_from_type_{gear_type}",
        )
        # Add constraints to calculate the sum of jewel emplacements
        model.Add(
            var_jewel_emplacement_sums
            == sum(_vars.jewel_emplacement_lists[gear_type][lvl])
        )
        # Store the sum variables in the _vars dictionary
        _vars.jewel_emplacement_sums[gear_type][lvl] = var_jewel_emplacement_sums


def _calculate_total_armor_jewel_emplacements(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    gear_types: list[str],
) -> None:
    """
    Calculate the total jewel emplacements for all armor pieces.

    This function iterates over each jewel level from 1 to 4 and creates a new integer variable
    representing the sum of jewel emplacements for all armor pieces at each level. It then adds
    constraints to calculate these sums across all armor pieces and stores the results in the
    _vars dictionary for later use.

    Args:
        model: The constraint programming model used for optimization.
        _vars: An instance of OptimizationVariables containing variables for the optimization process.
        gear_types: A list of unique armor piece identifiers used in the optimization.
    """

    for i in range(1, 5):
        # Define the jewel level string for clarity
        lvl = f"lvl{i}"

        # Create a new integer variable to represent the sum of jewel emplacements for all armor pieces at this level
        var_jewel_emplacement_sums = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"jewel_{lvl}_emplacement_sums_for_all_armor_pieces",
        )

        # Add a constraint to calculate the sum of jewel emplacements across all armor pieces for the current jewel level
        model.Add(
            var_jewel_emplacement_sums
            == sum(
                _vars.jewel_emplacement_sums[armor_piece][lvl]
                for armor_piece in gear_types
            )
        )

        # Store the calculated sum variable in the _vars dictionary for later use
        _vars.jewel_emplacement_sums_total_armor[lvl] = var_jewel_emplacement_sums


def _process_charms(model: cp_model.CpModel, _vars: OptimizationVariables) -> None:
    """
    Process charm data and add constraints to the model.

    This function iterates over each unique charm name, retrieves the corresponding charm data,
    and creates boolean and integer variables to represent the usage and talent levels of each charm.
    It adds constraints to enforce the talent levels based on whether the charm is used or not and
    ensures that only one charm can be equipped at a time.

    Args:
        model: The constraint programming model used for optimization.
        _vars: An instance of OptimizationVariables containing variables for the optimization process.
    """

    # Charm talents part
    # Get a sorted list of unique charm names
    unique_charm_name = charms["name"].unique().sort().to_list()

    # Iterate over each unique charm name
    for charm_name in unique_charm_name:
        # Filter charm data for the current charm name
        charm_data = charms.filter(pl.col("name") == charm_name).to_dicts()

        # Create a boolean variable to represent the usage of the charm
        use_charm_var = model.NewBoolVar(f"use_charm_{charm_name}")

        # Iterate over each row of charm data
        for row in charm_data:
            charm_name = row["name"]
            charm_talent = row["talent_name"]
            charm_lvl = row["talent_lvl"]

            # Create an integer variable for the talent level of the charm
            charm_talent_lvl = model.NewIntVar(
                lb=0,
                ub=30,
                name=f"charm_talent_{charm_talent}_lvl_{charm_lvl}_from_{charm_name}",
            )

            # Add constraints to enforce talent levels based on charm usage
            model.Add(charm_talent_lvl == charm_lvl).only_enforce_if(use_charm_var)
            model.Add(charm_talent_lvl == 0).only_enforce_if(use_charm_var.Not())

            # Append the talent level variable to the talent list
            _vars.talent_lists[charm_talent].append(charm_talent_lvl)

        # Store the boolean variable for charm usage
        _vars.use_charm_booleans[charm_name] = use_charm_var

    # Add the constraint to ensure only one charm can be equipped at a time
    model.Add(sum(_vars.use_charm_booleans.values()) <= 1)


def _set_weapon_talents(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    weapon: pl.DataFrame,
) -> None:
    """
    Set weapon talents in the constraint programming model.

    This function processes each weapon's talent data, creating integer variables
    for each talent level and adding constraints to the model to ensure the talent
    levels are correctly represented. The function also updates the talent list
    dictionary with these variables.

    Args:
        model: The constraint programming model used for optimization.
        _vars: An instance of OptimizationVariables containing variables for the optimization process.
        weapon: A DataFrame containing weapon data, including talent names and levels.
    """
    for row in weapon.to_dicts():
        # Extract talent name and level from the current row
        talent_name = row["talent_name"]
        talent_lvl = row["talent_lvl"]

        # Create an integer variable for the talent level of the weapon
        var_talent_lvl = model.NewIntVar(
            lb=1,
            ub=30,
            name=f"weapon_talent_{talent_name}_lvl_{talent_lvl}",
        )

        # Skip processing if the talent level is not defined
        if not talent_lvl:
            continue

        # Add a constraint to set the talent level variable to the actual talent level
        model.Add(var_talent_lvl == talent_lvl)

        # Register the variable in the talent list dictionary
        if _vars.talent_lists.get(talent_name) is None:
            _vars.talent_lists[talent_name] = []
        _vars.talent_lists[talent_name].append(var_talent_lvl)


def _create_weapon_jewel_slots(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    weapon: pl.DataFrame,
) -> None:
    """
    Create jewel slots for weapons in the constraint programming model.

    This function iterates over the jewel levels for a given weapon, creating integer
    variables for each jewel level and adding constraints to the model to ensure the
    jewel levels are correctly represented. The function also updates the jewel
    emplacement sums total for the weapon with these variables.

    Args:
        model: The constraint programming model used for optimization.
        _vars: An instance of OptimizationVariables containing variables for the optimization process.
        weapon: A DataFrame containing weapon data, including jewel levels.
    """
    row = weapon.to_dicts()[0]
    # Iterate over jewel levels from 1 to 3
    for i in range(1, 4):
        lvl = f"lvl{i}"

        # Create an integer variable for the jewel emplacement sums for the current level
        var_jewel_emplacement_sums = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"jewel_emplacement_sums_{lvl}_from_weapon",
        )

        # Add a constraint to set the jewel emplacement sums variable to the actual jewel level from the weapon
        model.Add(var_jewel_emplacement_sums == row[f"jewel_{lvl}"])

        # Register the variable in the jewel emplacement sums total for the weapon
        _vars.jewel_emplacement_sums_total_weapon[lvl] = var_jewel_emplacement_sums


def _register_jewel_usage(
    model: cp_model.CpModel,
    _vars: OptimizationVariables,
    all_jewels: pl.DataFrame,
    jewel_name: str,
) -> None:
    """
    Register the usage of a specific jewel in the constraint programming model.

    This function filters the jewel data to find entries matching the given jewel name,
    creates integer variables to represent the number of times the jewel is used, and
    adds constraints to ensure the total talent level is correctly calculated based on
    the jewel usage. It also updates the talent lists and jewel uses integers with the
    newly created variables.

    Args:
        model: The constraint programming model used for optimization.
        _vars: An instance of OptimizationVariables containing variables for the optimization process.
        all_jewels: A DataFrame containing all jewel data, including names and talent levels.
        jewel_name: The name of the jewel to register in the model.
    """
    # Filter the jewels data to get only the rows corresponding to the current jewel name
    jewel_data = all_jewels.filter(pl.col("jewel_name") == jewel_name)

    # Create an integer variable to represent the number of times the jewel is used
    nb_of_jewel_use = model.NewIntVar(lb=0, ub=100, name=f"nb_of_use_of_{jewel_name}")

    # Iterate over each row in the filtered jewel data
    for row in jewel_data.to_dicts():
        talent_name = row["talent_name"]
        talent_lvl = row["talent_lvl"]

        # Create an integer variable to represent the total talent level contributed by the jewel
        total_talent = model.NewIntVar(
            lb=0,
            ub=100,
            name=f"total_talent_{talent_name}_lvl_of_for_jewel_{jewel_name}",
        )

        # Add a constraint to ensure the total talent level is equal to the number of jewel uses times the talent level
        model.Add(total_talent == nb_of_jewel_use * talent_lvl)

        # Register the total talent variable in the talent lists dictionary
        if talent_name not in _vars.talent_lists:
            _vars.talent_lists[talent_name] = []
        _vars.talent_lists[talent_name].append(total_talent)

    # Get the jewel level from the current row
    jewel_lvl = row["jewel_lvl"]

    # Register the number of jewel uses in the jewel uses integers dictionary
    _vars.jewel_uses_integers[jewel_lvl][jewel_name] = nb_of_jewel_use


def solve(
    weapon_dict: dict,
    talent_list: list[dict],
    gear_types: list[str],
) -> dict:
    """
    Solve the armor optimization problem using constraint programming.

    Args:
        weapon_dict: Dictionary containing weapon information including name and jewel slots
        talent_list: List of dictionaries containing talent requirements with weights and target levels

    Returns:
        Dictionary containing the optimal solution with armor pieces, charm, weapon, and jewels
    """
    weapon_name = weapon_dict["name"]

    # Transform talent list into more usable dictionaries
    wanted_talents = {item["name"]: item["weight"] for item in talent_list}
    wanted_talents_objective = {
        item["name"]: item["target_level"] for item in talent_list
    }

    # Initialize constraint programming model
    model = cp_model.CpModel()

    # Initialize the optimization variables
    _vars = OptimizationVariables()

    # Get unique type of armor pieces
    gear_types = armor["piece"].unique().sort().to_list()
    # Get weapon data
    weapon = (
        weapons.filter(pl.col("name") == weapon_name)
        .explode("talents")
        .select(
            "name",
            *[
                pl.col("jewels").struct.field(str(i)).alias(f"jewel_lvl{i}")
                for i in range(4)
            ],
            pl.col("talents").struct.field("name").alias("talent_name"),
            pl.col("talents").struct.field("lvl").alias("talent_lvl"),
        )
    )
    # Get jewels data
    all_jewels = jewels.explode("jewel_talent_list").select(
        pl.col("name").alias("jewel_name"),
        "jewel_lvl",
        pl.col("jewel_talent_list").struct.field("name").alias("talent_name"),
        pl.col("jewel_talent_list").struct.field("lvl").alias("talent_lvl"),
    )
    # Get jewel types
    jewel_types = all_jewels.join(
        talents.select("group", pl.col("name").alias("talent_name")).unique(),
        on="talent_name",
    )

    # BEGIN CONSTRAINTS SETUP
    for gear_type in gear_types:
        # Add constraints related to the usage of armor piece and talent levels inherited from the armor pieces
        # TODO: Separate the variable indicating that the armor piece is equipped in an other function
        _process_armor_pieces(model=model, _vars=_vars, gear_type=gear_type)
        # Add constraints related to the amount of jewel emplacement and the size of the jewel emplacement
        _create_jewel_slots_for_armor_pieces(
            model=model, _vars=_vars, gear_type=gear_type
        )

    # Calculate the total jewel emplacement for all armor pieces for each jewel level
    _calculate_total_armor_jewel_emplacements(
        model=model, _vars=_vars, gear_types=gear_types
    )

    # Add constraints related to the usage of charms and the talent levels inherited from the charms
    # TODO: Separate the variable indicating that the charm is equipped in an other function
    _process_charms(model=model, _vars=_vars)

    # Add constraints related to the usage of weapon and the talent levels inherited from the weapon
    _set_weapon_talents(model=model, _vars=_vars, weapon=weapon)

    # Create variables that symbolizes the total number of weapon jewels
    _create_weapon_jewel_slots(model=model, _vars=_vars, weapon=weapon)

    # Jewels
    ## Register the number of jewels used

    unique_jewels_names = all_jewels["jewel_name"].unique().sort().to_list()
    for jewel_name in unique_jewels_names:
        _register_jewel_usage(
            model=model, _vars=_vars, jewel_name=jewel_name, all_jewels=all_jewels
        )

    # Add contraints for maximum number of jewels

    ## Get armor jewel types
    armor_jewel_types = jewel_types.filter(pl.col("group") == "Equip")

    # Jewels 3
    nb_of_uses_of_jewel3 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[3].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 3 can only fit in lvl 3
    model.Add(nb_of_uses_of_jewel3 <= _vars.jewel_emplacement_sums_total_armor["lvl3"])

    # Jewels 2
    nb_of_uses_of_jewel2 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[2].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 2 can fit in lvl 2, 3
    model.Add(
        nb_of_uses_of_jewel2
        <= sum(_vars.jewel_emplacement_sums_total_armor[f"lvl{i}"] for i in range(2, 4))
        - nb_of_uses_of_jewel3
    )
    # Jewels 1
    nb_of_uses_of_jewel1 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[1].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 1 can fit in lvl 1, 2, 3
    model.Add(
        nb_of_uses_of_jewel1
        <= sum(_vars.jewel_emplacement_sums_total_armor[f"lvl{i}"] for i in range(1, 4))
        - nb_of_uses_of_jewel3
        - nb_of_uses_of_jewel2
    )

    ## Get weapon jewel types
    weapon_jewel_types = jewel_types.filter(pl.col("group") == "Weapon")
    # Jewels 3
    nb_of_uses_of_jewel3 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[3].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 3 can only fit in lvl 3
    model.Add(nb_of_uses_of_jewel3 <= _vars.jewel_emplacement_sums_total_weapon["lvl3"])

    # Jewels 2
    nb_of_uses_of_jewel2 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[2].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 2 can fit in lvl 2, 3
    model.Add(
        nb_of_uses_of_jewel2
        <= sum(
            _vars.jewel_emplacement_sums_total_weapon[f"lvl{i}"] for i in range(2, 4)
        )
        - nb_of_uses_of_jewel3
    )

    # Jewels 1
    nb_of_uses_of_jewel1 = sum(
        var
        for jewel_name, var in _vars.jewel_uses_integers[1].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 1 can fit in lvl 1, 2, 3
    model.Add(
        nb_of_uses_of_jewel1
        <= sum(
            _vars.jewel_emplacement_sums_total_weapon[f"lvl{i}"] for i in range(1, 4)
        )
        - nb_of_uses_of_jewel3
        - nb_of_uses_of_jewel2
    )

    # Compute the talent sums
    for talent_name, talent_vars in _vars.talent_lists.items():
        var_talent_sum = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"talent_sum_for_{talent_name}",
        )
        model.Add(var_talent_sum == sum(talent_vars))

        _vars.talent_sums[talent_name] = var_talent_sum

    # Add talent sum cap
    unique_talent_names = talents["name"].unique().to_list()
    talents_lvl_max = (
        talents.explode("levels")
        .with_columns(
            pl.col("levels").struct.field("lvl").alias("talent_lvl"),
            pl.col("levels").struct.field("description").alias("talent_description"),
        )
        .filter(pl.col("talent_lvl") == pl.col("talent_lvl").max().over("name"))
        .select("group", "name", "talent_lvl")
    )
    for talent_name in unique_talent_names:
        # Get the talent max lvl
        individual_max_level = talents_lvl_max.filter(
            pl.col("name") == talent_name
        ).to_dicts()[0]

        var_talent_sum_capped = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"talent_sum_capped_for_{talent_name}",
        )
        model.AddMinEquality(
            target=var_talent_sum_capped,
            exprs=[
                _vars.talent_sums[talent_name],
                individual_max_level["talent_lvl"],
            ],
        )
        # Store the capped talent sum
        _vars.talent_sums_capped[talent_name] = var_talent_sum_capped

    # Set bonus talents
    group_talent_names = (
        talents.filter(pl.col("group") == "Group")["name"].unique().to_list()
    )
    for group_talent_name in group_talent_names:
        data = (
            talents
            #
            .filter(pl.col("name") == group_talent_name)
            .explode("levels")
            .with_columns(pl.col("levels").struct.field("lvl").alias("talent_lvl"))
        ).to_dicts()[0]
        # If the number of group talent is below the lvl, set it to 0
        group_has_enough_levels = model.NewBoolVar(
            name=f"group_talent_has_enough_levels_{group_talent_name}"
        )
        model.Add(
            _vars.talent_sums_capped[group_talent_name] >= data["talent_lvl"]
        ).OnlyEnforceIf(group_has_enough_levels)
        model.Add(
            _vars.talent_sums_capped[group_talent_name] < data["talent_lvl"]
        ).OnlyEnforceIf(group_has_enough_levels.Not())
        _vars.group_has_enough_level[group_talent_name] = group_has_enough_levels

    # Add set series bonus talents
    talent_series = (
        talents.filter(pl.col("group") == "Series")
        .explode("levels")
        .select(
            "name",
            pl.col("levels").struct.field("lvl").alias("talent_lvl"),
        )
    )
    talent_series_names = talent_series["name"].unique().to_list()
    for talent_series_name in talent_series_names:
        # Filter the talent series for the current series name
        unique_serie = talent_series.filter(pl.col("name") == talent_series_name)
        # Create a list of levels, starting with 0
        levels = [0] + unique_serie.sort("talent_lvl")["talent_lvl"].to_list() + [30]
        _vars.talent_series_interval[talent_series_name] = []
        # Iterate over pairs of consecutive levels
        for talent_inferior, talent_superior in pairwise(levels):
            # Create a boolean variable to check if the talent level is more than the inferior level
            var_talent_lvl_is_more_than_inferior = model.NewBoolVar(
                name=f"talent_series_{talent_series_name}_greater_than_{talent_inferior}_{talent_superior}"
            )
            # Add constraints based on the boolean variable
            model.Add(
                talent_inferior <= _vars.talent_sums_capped[talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_more_than_inferior)
            model.Add(
                talent_inferior > _vars.talent_sums_capped[talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_more_than_inferior.Not())

            # Create a boolean variable to check if the talent level is less than the superior level
            var_talent_lvl_is_less_than_superior = model.NewBoolVar(
                name=f"talent_series_{talent_series_name}_less_than_{talent_superior}_{talent_superior}"
            )
            # Add constraints based on the boolean variable
            model.Add(
                talent_superior > _vars.talent_sums_capped[talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_less_than_superior)
            model.Add(
                talent_superior <= _vars.talent_sums_capped[talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_less_than_superior.Not())

            # Create a boolean variable to check if the talent level is between the inferior and superior levels
            var_talent_lvl_is_between = model.NewBoolVar(
                name=f"talent_series_{talent_series_name}_between_{talent_inferior}_and_{talent_superior}"
            )
            # Add constraints to enforce the 'between' condition
            model.Add(
                sum(
                    [
                        var_talent_lvl_is_more_than_inferior,
                        var_talent_lvl_is_less_than_superior,
                    ]
                )
                == 2
            ).only_enforce_if(var_talent_lvl_is_between)
            model.Add(
                sum(
                    [
                        var_talent_lvl_is_more_than_inferior,
                        var_talent_lvl_is_less_than_superior,
                    ]
                )
                < 2
            ).only_enforce_if(var_talent_lvl_is_between.Not())

            # Add a constraint to set the talent level to the inferior level if the 'between' condition is met
            var = model.NewIntVar(
                name=f"talent_sums_final_{talent_series_name}_interrval_{talent_inferior}_{talent_superior}",
                lb=0,
                ub=30,
            )
            _vars.talent_series_interval[talent_series_name].append(var)
            model.Add(var == talent_inferior).only_enforce_if(var_talent_lvl_is_between)
            model.Add(var == 0).only_enforce_if(var_talent_lvl_is_between.Not())

    ## Objective
    for name, var in _vars.talent_sums_capped.items():
        if name not in _vars.talent_sums_final and name not in talent_series_names:
            newvar = model.NewIntVar(name=f"talent_sums_final_{name}", lb=0, ub=30)
            _vars.talent_sums_final[name] = newvar
            if name in group_talent_names:
                model.Add(_vars.talent_sums_final[name] == var).only_enforce_if(
                    _vars.group_has_enough_level[name]
                )
                model.Add(_vars.talent_sums_final[name] == 0).only_enforce_if(
                    _vars.group_has_enough_level[name].Not()
                )
            else:
                model.Add(_vars.talent_sums_final[name] == var)
    for name, var_list in _vars.talent_series_interval.items():
        if name not in _vars.talent_sums_final:
            newvar = model.NewIntVar(name=f"talent_sums_final_{name}", lb=0, ub=30)
            _vars.talent_sums_final[name] = newvar
            model.Add(_vars.talent_sums_final[name] == sum(var_list))

    # Add additionnal value for free jewels emplacements
    lvl_emplacements = {}
    for lvl in range(1, 4):
        lvl_emplacements[lvl] = []
        for gear_type, _dict in _vars.jewel_emplacement_sums.items():
            lvl_emplacements[lvl].append(_dict[f"lvl{lvl}"])

    maximize_nb_of_free_jewels = []
    maximize_nb_of_free_jewels.extend(
        (sum(lvl_emplacements[lvl]) - sum(_vars.jewel_uses_integers[lvl].values()))
        * 10**3
        * 10**lvl
        for lvl in range(1, 4)
    )
    # Add additional objective value for eventual addional talents (still minimize nb of jewels)
    nb_of_talents = sum(_vars.talent_sums_final.values())

    # Add optional strict talent optimization
    talent_objective = (
        sum(
            _vars.talent_sums_final[talent] * 10**talent_weight
            for talent, talent_weight in wanted_talents.items()
        )
        * 10**9
    )
    # Add a penalty if talent lvl is greater than the objective lvl
    abs_diff_vars = []
    for talent_name, talent_var in _vars.talent_sums_final.items():
        if talent_name in wanted_talents_objective.keys():
            objective_lvl = wanted_talents_objective[talent_name]
            if objective_lvl == -1:
                continue
            abs_diff = model.NewIntVar(name=f"abs_diff_{talent_name}", lb=0, ub=100)
            model.AddAbsEquality(abs_diff, talent_var - objective_lvl)
            abs_diff_vars.append(abs_diff * 10 ** wanted_talents[talent_name] * 10**9)

    model.maximize(
        sum(
            [
                talent_objective,
                nb_of_talents,
                sum(maximize_nb_of_free_jewels),
                -sum(abs_diff_vars),
            ]
        )
    )
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    solver_statuses = {
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }

    # ic(f"Solver status: {solver_statuses[status]}")
    solution = {"jewels": {}, "weapon": weapon.to_dicts()[0]["name"]}
    for gear_type, var_dict in _vars.use_armor_piece_booleans.items():
        for name, var in var_dict.items():
            if solver.value(var) == 1:
                solution[gear_type] = name
    for charm, var in _vars.use_charm_booleans.items():
        if solver.value(expression=var) == 1:
            solution["charm"] = charm
            # display(charms.filter(pl.col("name") == charm))
    for jewel_lvl, var_dict in _vars.jewel_uses_integers.items():
        for jewel_name, var in var_dict.items():
            if solver.value(var) > 0:
                solution["jewels"][jewel_name] = solver.value(var)

    return solution


def generate_markdown_from_solution(solution: dict) -> str:
    """
    Generate a markdown representation from the given solution.

    Args:
        solution (dict[str, dict[str, int]]): A dictionary containing the solution details,
            including jewels, weapon, and charm.

    Returns:
        str: A markdown formatted string representing the solution.
    """

    solution_jewels = solution["jewels"]
    weapon_name = solution["weapon"]
    charm_name = solution["charm"]

    return (
        (
            # Create a DataFrame from the solution dictionary for armor pieces
            df := pl.DataFrame(
                [
                    {"name": solution[_type], "type": _type, "order": order}
                    for order, _type in enumerate(
                        ["Tête", "Torse", "Bras", "Taille", "Jambes"]
                    )
                ],
            )
            # Join with armor DataFrame to get talent levels
            .join(
                armor,
                on="name",
                how="inner",
            )
            .sort("order")
            # Pivot to aggregate talent levels by name
            .pivot(
                index="talent_name",
                on="name",
                values="talent_level",
                aggregate_function="sum",
            )
            # Join with jewels DataFrame to include jewel talents
            .join(
                pl.DataFrame(
                    [
                        {"name": key, "quantity": value}
                        for key, value in solution_jewels.items()
                    ]
                    if len(solution_jewels.keys()) > 0
                    else {"name": "", "quantity": 0}
                )
                .join(jewels, on="name", how="inner")
                .explode("jewel_talent_list")
                .with_columns(
                    pl.col("jewel_talent_list")
                    .struct.field("name")
                    .alias("talent_name"),
                    pl.col("jewel_talent_list").struct.field("lvl").alias("talent_lvl"),
                )
                .with_columns(
                    (pl.col("talent_lvl") * pl.col("quantity")).alias("Joyaux"),
                )
                .select("talent_name", "Joyaux"),
                on="talent_name",
                how="outer",
            )
            # Handle missing talent names
            .with_columns(pl.coalesce("talent_name", "talent_name_right"))
            .drop("talent_name_right")
            # Join with weapon talents
            .join(
                weapons
                # Filter for the specific weapon and explode talents
                .filter(pl.col("name") == weapon_name)
                .explode("talents")
                .select(
                    pl.col("talents").struct.field("name").alias("talent_name"),
                    pl.col("talents").struct.field("lvl").alias(weapon_name),
                ),
                on="talent_name",
                how="outer",
            )
            # Handle missing talent names
            .with_columns(pl.coalesce("talent_name", "talent_name_right"))
            .drop("talent_name_right")
            # Join with charm talents
            .join(
                charms.filter(pl.col("name") == charm_name)
                .select("talent_name", "talent_lvl")
                .rename({"talent_lvl": charm_name}),
                on="talent_name",
                how="outer",
            )
            # Handle missing talent names
            .with_columns(pl.coalesce("talent_name", "talent_name_right"))
            .drop("talent_name_right")
        )
        # Initialize Total column with zeros
        .with_columns(pl.lit(0).alias("Total"))
        .fill_null(0)
        # Calculate the total talent level for each talent
        .with_columns(
            pl.struct(df.columns[1:])
            .alias("Total")
            .map_elements(lambda s: sum(s.values()))
            .alias("Total")
        )
        .sort("Total")
        # Join with talents DataFrame to get descriptions
        .join_asof(
            talents.explode("levels")
            .select(
                pl.col("name").alias("talent_name"),
                pl.col("levels").struct.field("lvl").alias("Total"),
                pl.col("levels").struct.field("description").alias("Description"),
            )
            .sort("Total"),
            by="talent_name",
            on="Total",
        )
        # Convert all columns to strings and remove zeros
        .with_columns(pl.col(col).cast(pl.String).alias(col) for col in df.columns)
        .with_columns(pl.col(col).str.replace("0", "") for col in df.columns)
        .with_columns(pl.col("Joyaux").str.replace("0", ""))
        # Sort by Total and talent name
        .sort("Total", "talent_name", descending=[True, False])
        .rename({"talent_name": "Talent"})
        # Convert to pandas DataFrame and then to markdown
        .to_pandas()
        .to_markdown(index=False)
    )


def generate_markdown_for_jewels(solution: dict) -> list[str]:
    """
    Generate a markdown representation of jewel data from the given solution.

    This function processes the solution dictionary to extract jewel types and their
    associated talent groups, prepares armor slots data, and weapon slots data.
    It returns a list of strings in markdown format representing the jewel information.

    Args:
        solution (dict): A dictionary containing the solution data with keys for jewels,
                            armor types, and weapon names.

    Returns:
        list[str]: A list of strings formatted in markdown representing the jewel data.
    """

    solution_jewels = solution["jewels"]
    # Extract jewel types and their associated talent groups
    jewel_types = (
        jewels
        #
        .explode("jewel_talent_list")
        .select(
            "name",
            pl.col("jewel_talent_list").struct.field("name").alias("jewel_talent_name"),
        )
        .join(
            talents.select(pl.col("name").alias("jewel_talent_name"), "group").unique(),
            on="jewel_talent_name",
            how="left",
        )
    )

    # Prepare armor slots data from the solution
    armor_slots = (
        pl.DataFrame(
            [
                {"name": solution[_type], "type": _type, "order": order}
                for order, _type in enumerate(
                    ["Tête", "Torse", "Bras", "Taille", "Jambes"]
                )
            ],
        )
        #
        .join(
            armor,
            on="name",
            how="inner",
        )
        .unique("name")
        .select(pl.col(f"jewel_{i}").sum() for i in range(1, 4))
        .unpivot()
        .sort("variable")
        .to_dicts()
    )

    # Prepare weapon slots data based on the weapon name
    weapon_slots = (
        weapons
        #
        .filter(pl.col("name") == weapon_name)
        .select(
            pl.col("jewels").struct.field(str(i)).alias(f"jewel_{str(i)}")
            for i in range(1, 4)
        )
        .unpivot()
        .sort("variable")
        .to_dicts()
    )

    # Generate markdown for armor jewels
    armor_md = "# **Joyaux armure** \n\n"
    for row in armor_slots:
        size = row["variable"].replace("jewel_", "")
        at_least_one_jewel = False
        free_slots = row["value"]
        for name, quantity in solution_jewels.items():
            # Skip jewels that belong to weapons
            if (
                name
                in jewel_types.filter(pl.col("group") == "Weapon")["name"].to_list()
            ):
                continue
            # Add header for jewel size if not already added
            if (
                not at_least_one_jewel
                and quantity > 0
                and name.split("[")[-1].replace("]", "") == size
            ):
                armor_md += f"### **Joyaux taille {size}**,  slots: {free_slots}\n\n"
                at_least_one_jewel = True
            # Add jewel details
            if name.split("[")[-1].replace("]", "") == size:
                armor_md += f"* {name} x{quantity}\n\n"
                free_slots -= quantity
        # Add free slots information if any slots are left
        if free_slots > 0:
            if not at_least_one_jewel:
                armor_md += f"### **Joyaux taille {size}**,  slots: {free_slots}\n\n"
            armor_md += f"* *Libre: {free_slots}*\n\n"

    # Generate markdown for weapon jewels
    weapon_md = "# **Joyaux arme** \n\n"
    for row in weapon_slots:
        size = row["variable"].replace("jewel_", "")
        at_least_one_jewel = False
        free_slots = row["value"]
        for name, quantity in solution_jewels.items():
            # Skip jewels that do not belong to weapons
            if (
                name
                not in jewel_types.filter(pl.col("group") == "Weapon")["name"].to_list()
            ):
                continue
            # Add header for jewel size if not already added
            if (
                not at_least_one_jewel
                and quantity > 0
                and name.split("[")[-1].replace("]", "") == size
            ):
                weapon_md += f"### **Joyaux taille {size}**,  slots: {free_slots}\n\n"
                at_least_one_jewel = True
            else:
                continue
            # Add jewel details
            if name.split("[")[-1].replace("]", "") == size:
                weapon_md += f"* {name} x{quantity}\n\n"
                free_slots -= quantity
        # Add free slots information if any slots are left
        if free_slots > 0:
            if not at_least_one_jewel:
                weapon_md += f"### **Joyaux taille {size}**,  slots: {free_slots}\n\n"
            weapon_md += f"* *Libre: {free_slots}*\n\n"
    return armor_md, weapon_md


if __name__ == "__main__":
    # weapon_name = {"name": "Lame d'espoir"}
    weapon_name = weapons.filter(pl.col("name") == "Lame d'espoir").to_dicts()[0]
    talent_list = [
        {"name": "Tyrannie du Gore Magala", "target_level": 2, "weight": 1},
        {"name": "Volonté de l'Anjanath tonnerre", "target_level": 2, "weight": 1},
    ]
    print(solve(weapon_name, talent_list))

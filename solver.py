from collections import defaultdict
from itertools import pairwise

import polars as pl
from icecream import ic
from ortools.sat.python import cp_model
from rich import traceback

from dataclasses import dataclass, field
from typing import Dict, List, Any

traceback.install()

armor = pl.read_parquet("data/armor_pieces.parquet")
charms = pl.read_parquet("data/charms.parquet")
jewels = pl.read_parquet("data/jewels.parquet")
talents = pl.read_parquet("data/talents.parquet")
weapons = pl.read_parquet("data/weapons.parquet")


@dataclass
class OptimizationVariables:
    use_armor_piece_booleans: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    use_charm_booleans: Dict[str, Any] = field(default_factory=dict)
    talent_lists: Dict[str, List[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    talent_sums: Dict[str, Any] = field(default_factory=dict)
    talent_sums_capped: Dict[str, Any] = field(default_factory=dict)
    talent_series_interval: Dict[str, List[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    talent_sums_final: Dict[str, Any] = field(default_factory=dict)
    group_has_enough_level: Dict[str, Any] = field(default_factory=dict)
    jewel_emplacement_lists: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    jewel_emplacement_sums: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    jewel_emplacement_sums_total_armor: Dict[str, Any] = field(default_factory=dict)
    jewel_emplacement_sums_total_weapon: Dict[str, Any] = field(default_factory=dict)
    jewel_uses_integers: Dict[int, Dict[str, Any]] = field(
        default_factory=lambda: defaultdict(dict)
    )


def _process_armor_pieces(
    model: cp_model.CpModel, _vars: OptimizationVariables, unique_pieces: list
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
        unique_pieces (list): A list of unique armor pieces to process.

    Returns:
        None
    """
    for armor_piece in unique_pieces:
        # _vars.setdefault("use_armor_piece_booleans", {}).setdefault(armor_piece, {})
        # _vars.setdefault("jewel_emplacement_lists", {}).setdefault(armor_piece, {})
        # _vars.setdefault("jewel_emplacement_sums", {}).setdefault(armor_piece, {})

        armor_piece_filtered = armor.filter(pl.col("piece") == armor_piece)
        unique_armor_pieces_names = (
            armor_piece_filtered["name"].unique().sort().to_list()
        )
        for unique_armor_piece_name in unique_armor_pieces_names:
            unique_armor_piece = armor_piece_filtered.filter(
                pl.col("name") == unique_armor_piece_name
            )

            # Define a boolean that tells if the armor piece is equipped
            names = unique_armor_piece["name"].to_list()
            name = names[0]
            armor_piece_equipped = model.NewBoolVar(f"use_piece_{armor_piece}_{name}")

            for row in unique_armor_piece.iter_rows():
                (name, talent_name, talent_level) = row[1:4]
                # Create a variable that tells that the talent is active due to the fact that the armor piece is equipped
                var_talent_lvl = model.NewIntVar(
                    lb=0,
                    ub=30,
                    name=f"talent_{talent_name}_from_type_{armor_piece}_with_{name}",
                )
                model.Add(var_talent_lvl == talent_level).only_enforce_if(
                    armor_piece_equipped
                )
                model.Add(var_talent_lvl == 0).only_enforce_if(
                    armor_piece_equipped.Not()
                )

                # # Create the key if it doesn't exist
                # if _vars.talent_lists.get(talent_name) is None:
                #     _vars.talent_lists[talent_name] = []

                # Register variables
                _vars.talent_lists[talent_name].append(var_talent_lvl)

            # Store the boolean that tells if the armor piece is equipped
            _vars.use_armor_piece_booleans[armor_piece][name] = armor_piece_equipped


def solve(
    weapon_dict: dict,
    talent_list: list[dict],
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

    # Organize variables in a structured dictionary
    _vars = {
        "group_has_enough_level": {},  # Tracks if a group of talents meets the required level
        "jewel_emplacement_lists": {},  # Stores lists of jewel placements for each armor piece
        "jewel_emplacement_sums_total_armor": {},  # Sums of jewel levels for all armor pieces
        "jewel_emplacement_sums_total_weapon": {},  # Sums of jewel levels for the weapon
        "jewel_emplacement_sums": {},  # Sums of jewel levels for each armor piece
        "jewel_uses_integers": {},  # Tracks the integer usage of jewels
        "talent_lists": {},  # Stores lists of talents for each armor piece
        "talent_series_interval": {},  # Defines intervals for talent series
        "talent_sums_capped": {},  # Capped sums of talent levels
        "talent_sums_final": {},  # Final sums of talent levels after adjustments
        "talent_sums": {},  # Sums of talent levels for each armor piece
        "use_armor_piece_booleans": {},  # Boolean flags indicating if an armor piece is used
        "use_charm_booleans": {},  # Boolean flags indicating if a charm is used
    }
    _vars = OptimizationVariables()

    unique_pieces = armor["piece"].unique().sort().to_list()

    _process_armor_pieces(model=model, _vars=_vars, unique_pieces=unique_pieces)
    for armor_piece in unique_pieces:
        armor_piece_filtered = armor.filter(pl.col("piece") == armor_piece)
        unique_armor_pieces_names = (
            armor_piece_filtered["name"].unique().sort().to_list()
        )
        # Jewel emplacement part
        for unique_armor_piece_name in unique_armor_pieces_names:
            unique_armor_piece = armor_piece_filtered.filter(
                pl.col("name") == unique_armor_piece_name
            )

            # Define a boolean that tells if the armor piece is equipped
            names = unique_armor_piece["name"].to_list()
            name = names[0]
            armor_piece_equipped = model.NewBoolVar(f"use_piece_{armor_piece}_{name}")

            for row in unique_armor_piece.iter_rows():
                (
                    _,
                    name,
                    talent_name,
                    talent_level,
                    jewel_0,
                    jewel_1,
                    jewel_2,
                    jewel_3,
                    jewel_4,
                ) = row

                # Jewel emplacement part
                # LVL 1
                var_nb_jewel_1 = model.NewIntVar(
                    lb=0,
                    ub=4,
                    name=f"jewel_lvl_1_from_type_{armor_piece}_with_{name}",
                )
                model.Add(var_nb_jewel_1 == jewel_1).only_enforce_if(
                    armor_piece_equipped
                )
                model.Add(var_nb_jewel_1 == 0).only_enforce_if(
                    armor_piece_equipped.Not()
                )

                # LVL 2
                var_nb_jewel_2 = model.NewIntVar(
                    lb=0,
                    ub=4,
                    name=f"jewel_lvl_2_from_type_{armor_piece}_with_{name}",
                )
                model.Add(var_nb_jewel_2 == jewel_2).only_enforce_if(
                    armor_piece_equipped
                )
                model.Add(var_nb_jewel_2 == 0).only_enforce_if(
                    armor_piece_equipped.Not()
                )

                # LVL 3
                var_nb_jewel_3 = model.NewIntVar(
                    lb=0,
                    ub=4,
                    name=f"jewel_lvl_3_from_type_{armor_piece}_with_{name}",
                )
                model.Add(var_nb_jewel_3 == jewel_3).only_enforce_if(
                    armor_piece_equipped
                )
                model.Add(var_nb_jewel_3 == 0).only_enforce_if(
                    armor_piece_equipped.Not()
                )

                # LVL 4
                var_nb_jewel_4 = model.NewIntVar(
                    lb=0,
                    ub=4,
                    name=f"jewel_lvl_4_from_type_{armor_piece}_with_{name}",
                )
                model.Add(var_nb_jewel_4 == jewel_4).only_enforce_if(
                    armor_piece_equipped
                )
                model.Add(var_nb_jewel_4 == 0).only_enforce_if(
                    armor_piece_equipped.Not()
                )

            # if _vars.get("jewel_emplacement_lists").get(armor_piece) is None:
            #     _vars.jewel_emplacement_lists[armor_piece] = {}

            # if (
            #     _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl1")
            #     is None
            # ):
            #     _vars.jewel_emplacement_lists[armor_piece]["lvl1"] = []
            _vars.jewel_emplacement_lists[armor_piece]["lvl1"].append(var_nb_jewel_1)

            # if (
            #     _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl2")
            #     is None
            # ):
            #     _vars.jewel_emplacement_lists[armor_piece]["lvl2"] = []
            _vars.jewel_emplacement_lists[armor_piece]["lvl2"].append(var_nb_jewel_2)

            # if (
            #     _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl3")
            #     is None
            # ):
            #     _vars.jewel_emplacement_lists[armor_piece]["lvl3"] = []
            _vars.jewel_emplacement_lists[armor_piece]["lvl3"].append(var_nb_jewel_3)

            # if (
            #     _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl4")
            #     is None
            # ):
            #     _vars.jewel_emplacement_lists[armor_piece]["lvl4"] = []
            _vars.jewel_emplacement_lists[armor_piece]["lvl4"].append(var_nb_jewel_4)

        # Create variables that are the sum of the jewel emplacements
        for i in range(1, 5):
            lvl = f"lvl{i}"
            var_jewel_emplacement_sums = model.NewIntVar(
                lb=0,
                ub=30,
                name=f"jewel_emplacement_sums_{lvl}_from_type_{armor_piece}",
            )
            model.Add(
                var_jewel_emplacement_sums
                == sum(_vars.jewel_emplacement_lists[armor_piece][lvl])
            )
            _vars.jewel_emplacement_sums[armor_piece][lvl] = var_jewel_emplacement_sums

        # Add the constraint of only one type of armor piece equipped at a time
        model.Add(sum(_vars.use_armor_piece_booleans[armor_piece].values()) <= 1)

    # Create a variable that symbolizes the total number of armor jewels
    for i in range(1, 5):
        lvl = f"lvl{i}"
        var_jewel_emplacement_sums = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"jewel_{lvl}_emplacement_sums_for_all_armor_pieces",
        )
        model.Add(
            var_jewel_emplacement_sums
            == sum(
                _vars.jewel_emplacement_sums[armor_piece][lvl]
                for armor_piece in unique_pieces
            )
        )
        _vars.jewel_emplacement_sums_total_armor[lvl] = var_jewel_emplacement_sums

    # Charm talents part
    unique_charm_name = charms["name"].unique().sort().to_list()
    for charm_name in unique_charm_name:
        charm_data = charms.filter(pl.col("name") == charm_name).to_dicts()

        use_charm_var = model.NewBoolVar(f"use_charm_{charm_name}")
        for row in charm_data:
            charm_name = row["name"]
            charm_talent = row["talent_name"]
            charm_lvl = row["talent_lvl"]

            charm_talent_lvl = model.NewIntVar(
                lb=0,
                ub=30,
                name=f"charm_talent_{talent_name}_lvl_{charm_lvl}_from_{charm_name}",
            )
            model.Add(charm_talent_lvl == charm_lvl).only_enforce_if(use_charm_var)
            model.Add(charm_talent_lvl == 0).only_enforce_if(use_charm_var.Not())

            # Register the variable
            # if _vars.talent_lists.get(charm_talent) is None:
            #     _vars.talent_lists[charm_talent] = []

            _vars.talent_lists[charm_talent].append(charm_talent_lvl)
        _vars.use_charm_booleans[charm_name] = use_charm_var

    # Add the constraint of only one charm equipped at a time
    model.Add(sum(_vars.use_charm_booleans.values()) <= 1)

    # Weapon talent part
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
    for row in weapon.to_dicts():
        talent_name = row["talent_name"]
        talent_lvl = row["talent_lvl"]
        var_talent_lvl = model.NewIntVar(
            lb=1,
            ub=30,
            name=f"weapon_talent_{talent_name}_lvl_{talent_lvl}",
        )
        if not talent_lvl:
            continue
        model.Add(var_talent_lvl == talent_lvl)
        # Register the variable
        if _vars.talent_lists.get(talent_name) is None:
            _vars.talent_lists[talent_name] = []
        _vars.talent_lists[talent_name].append(var_talent_lvl)

    # Create a variable that symbolizes the total number of weapon jewels
    for i in range(1, 4):
        lvl = f"lvl{i}"
        var_jewel_emplacement_sums = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"jewel_emplacement_sums_{lvl}_from_weapon",
        )
        model.Add(var_jewel_emplacement_sums == row[f"jewel_{lvl}"])
        # Register the variable
        _vars.jewel_emplacement_sums_total_weapon[lvl] = var_jewel_emplacement_sums

    # Jewels
    ## Register the number of jewels used
    all_jewels = jewels.explode("jewel_talent_list").select(
        pl.col("name").alias("jewel_name"),
        "jewel_lvl",
        pl.col("jewel_talent_list").struct.field("name").alias("talent_name"),
        pl.col("jewel_talent_list").struct.field("lvl").alias("talent_lvl"),
    )
    unique_jewels_names = all_jewels["jewel_name"].unique().sort().to_list()
    for jewel_name in unique_jewels_names:
        jewel_data = all_jewels.filter(pl.col("jewel_name") == jewel_name)
        nb_of_jewel_use = model.NewIntVar(
            lb=0, ub=100, name=f"nb_of_use_of_{jewel_name}"
        )
        for row in jewel_data.to_dicts():
            talent_name = row["talent_name"]
            talent_lvl = row["talent_lvl"]
            total_talent = model.NewIntVar(
                lb=0,
                ub=100,
                name=f"total_talent_{talent_name}_lvl_of_for_jewel_{jewel_name}",
            )
            model.Add(total_talent == nb_of_jewel_use * talent_lvl)

            if talent_name not in _vars.talent_lists:
                _vars.talent_lists[talent_name] = []
            _vars.talent_lists[talent_name].append(total_talent)
        jewel_lvl = row["jewel_lvl"]
        # if jewel_lvl not in _vars.jewel_uses_integers:
        #     _vars.jewel_uses_integers[jewel_lvl] = {}
        _vars.jewel_uses_integers[jewel_lvl][jewel_name] = nb_of_jewel_use

    # Add contraints for maximum number of jewels
    ## Get jewel types
    jewel_types = all_jewels.join(
        talents.select("group", pl.col("name").alias("talent_name")).unique(),
        on="talent_name",
    )

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
        for armor_piece, _dict in _vars.jewel_emplacement_sums.items():
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
    for armor_piece, var_dict in _vars.use_armor_piece_booleans.items():
        for name, var in var_dict.items():
            if solver.value(var) == 1:
                solution[armor_piece] = name
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

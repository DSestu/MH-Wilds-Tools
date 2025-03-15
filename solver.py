from collections import defaultdict
from itertools import pairwise

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

weapon_name = "Brisefoi Leibolkule"


def solve(
    weapon_dict: dict,
    talent_list: list[dict],
) -> dict:
    weapon_name = weapon_dict["name"]
    ic(weapon_name)
    ic(talent_list)
    wanted_talents = {item["name"]: item["weight"] for item in talent_list}
    wanted_talents_objective = {
        item["name"]: item["target_level"] for item in talent_list
    }
    ic(wanted_talents)
    model = cp_model.CpModel()
    _vars = {
        "use_armor_piece_booleans": {},
        "use_charm_booleans": {},
        "talent_lists": {},
        "talent_sums": {},
        "talent_sums_capped": {},
        "talent_series_interval": {},
        "talent_sums_final": {},
        "group_has_enough_level": {},
        "jewel_emplacement_lists": {},
        "jewel_emplacement_sums": {},
        "jewel_emplacement_sums_total_armor": {},
        "jewel_emplacement_sums_total_weapon": {},
        "jewel_uses_integers": {},
    }

    unique_pieces = armor["piece"].unique().sort().to_list()

    for armor_piece in unique_pieces:
        if _vars.get("use_armor_piece_booleans").get(armor_piece) is None:
            _vars["use_armor_piece_booleans"][armor_piece] = {}

        if _vars.get("jewel_emplacement_lists").get(armor_piece) is None:
            _vars["jewel_emplacement_lists"][armor_piece] = {}

        if _vars.get("jewel_emplacement_sums").get(armor_piece) is None:
            _vars["jewel_emplacement_sums"][armor_piece] = {}

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

                # Create the key if it doesn't exist
                if _vars["talent_lists"].get(talent_name) is None:
                    _vars["talent_lists"][talent_name] = []

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

                # Register variables
                _vars["talent_lists"][talent_name].append(var_talent_lvl)

            if _vars.get("jewel_emplacement_lists").get(armor_piece) is None:
                _vars["jewel_emplacement_lists"][armor_piece] = {}

            if (
                _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl1")
                is None
            ):
                _vars["jewel_emplacement_lists"][armor_piece]["lvl1"] = []
            _vars["jewel_emplacement_lists"][armor_piece]["lvl1"].append(var_nb_jewel_1)

            if (
                _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl2")
                is None
            ):
                _vars["jewel_emplacement_lists"][armor_piece]["lvl2"] = []
            _vars["jewel_emplacement_lists"][armor_piece]["lvl2"].append(var_nb_jewel_2)

            if (
                _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl3")
                is None
            ):
                _vars["jewel_emplacement_lists"][armor_piece]["lvl3"] = []
            _vars["jewel_emplacement_lists"][armor_piece]["lvl3"].append(var_nb_jewel_3)

            if (
                _vars.get("jewel_emplacement_lists").get(armor_piece).get("lvl4")
                is None
            ):
                _vars["jewel_emplacement_lists"][armor_piece]["lvl4"] = []
            _vars["jewel_emplacement_lists"][armor_piece]["lvl4"].append(var_nb_jewel_4)

            # Store the boolean that tells if the armor piece is equipped
            _vars["use_armor_piece_booleans"][armor_piece][name] = armor_piece_equipped

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
                == sum(_vars["jewel_emplacement_lists"][armor_piece][lvl])
            )
            _vars["jewel_emplacement_sums"][armor_piece][lvl] = (
                var_jewel_emplacement_sums
            )

        # Add the constraint of only one type of armor piece equipped at a time
        model.Add(sum(_vars["use_armor_piece_booleans"][armor_piece].values()) <= 1)

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
                _vars["jewel_emplacement_sums"][armor_piece][lvl]
                for armor_piece in unique_pieces
            )
        )
        _vars["jewel_emplacement_sums_total_armor"][lvl] = var_jewel_emplacement_sums

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
            if _vars["talent_lists"].get(charm_talent) is None:
                _vars["talent_lists"][charm_talent] = []

            _vars["talent_lists"][charm_talent].append(charm_talent_lvl)
        _vars["use_charm_booleans"][charm_name] = use_charm_var

    # Add the constraint of only one charm equipped at a time
    model.Add(sum(_vars["use_charm_booleans"].values()) <= 1)

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
        if _vars["talent_lists"].get(talent_name) is None:
            _vars["talent_lists"][talent_name] = []
        _vars["talent_lists"][talent_name].append(var_talent_lvl)

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
        _vars["jewel_emplacement_sums_total_weapon"][lvl] = var_jewel_emplacement_sums

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

            if talent_name not in _vars["talent_lists"]:
                _vars["talent_lists"][talent_name] = []
            _vars["talent_lists"][talent_name].append(total_talent)
        jewel_lvl = row["jewel_lvl"]
        if jewel_lvl not in _vars["jewel_uses_integers"]:
            _vars["jewel_uses_integers"][jewel_lvl] = {}
        _vars["jewel_uses_integers"][jewel_lvl][jewel_name] = nb_of_jewel_use

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
        for jewel_name, var in _vars["jewel_uses_integers"][3].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 3 can only fit in lvl 3
    model.Add(
        nb_of_uses_of_jewel3 <= _vars["jewel_emplacement_sums_total_armor"]["lvl3"]
    )

    # Jewels 2
    nb_of_uses_of_jewel2 = sum(
        var
        for jewel_name, var in _vars["jewel_uses_integers"][2].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 2 can fit in lvl 2, 3
    model.Add(
        nb_of_uses_of_jewel2
        <= sum(
            _vars["jewel_emplacement_sums_total_armor"][f"lvl{i}"] for i in range(2, 4)
        )
        - nb_of_uses_of_jewel3
    )
    # Jewels 1
    nb_of_uses_of_jewel1 = sum(
        var
        for jewel_name, var in _vars["jewel_uses_integers"][1].items()
        if jewel_name in armor_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 1 can fit in lvl 1, 2, 3
    model.Add(
        nb_of_uses_of_jewel1
        <= sum(
            _vars["jewel_emplacement_sums_total_armor"][f"lvl{i}"] for i in range(1, 4)
        )
        - nb_of_uses_of_jewel3
        - nb_of_uses_of_jewel2
    )

    ## Get weapon jewel types
    weapon_jewel_types = jewel_types.filter(pl.col("group") == "Weapon")
    # Jewels 3
    nb_of_uses_of_jewel3 = sum(
        var
        for jewel_name, var in _vars["jewel_uses_integers"][3].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 3 can only fit in lvl 3
    model.Add(
        nb_of_uses_of_jewel3 <= _vars["jewel_emplacement_sums_total_weapon"]["lvl3"]
    )

    # Jewels 2
    nb_of_uses_of_jewel2 = sum(
        var
        for jewel_name, var in _vars["jewel_uses_integers"][2].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 2 can fit in lvl 2, 3
    model.Add(
        nb_of_uses_of_jewel2
        <= sum(
            _vars["jewel_emplacement_sums_total_weapon"][f"lvl{i}"] for i in range(2, 4)
        )
        - nb_of_uses_of_jewel3
    )

    # Jewels 1
    nb_of_uses_of_jewel1 = sum(
        var
        for jewel_name, var in _vars["jewel_uses_integers"][1].items()
        if jewel_name in weapon_jewel_types["jewel_name"].unique().to_list()
    )
    # Jewels 1 can fit in lvl 1, 2, 3
    model.Add(
        nb_of_uses_of_jewel1
        <= sum(
            _vars["jewel_emplacement_sums_total_weapon"][f"lvl{i}"] for i in range(1, 4)
        )
        - nb_of_uses_of_jewel3
        - nb_of_uses_of_jewel2
    )

    # Compute the talent sums
    for talent_name, talent_vars in _vars["talent_lists"].items():
        var_talent_sum = model.NewIntVar(
            lb=0,
            ub=30,
            name=f"talent_sum_for_{talent_name}",
        )
        model.Add(var_talent_sum == sum(talent_vars))

        _vars["talent_sums"][talent_name] = var_talent_sum

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
                _vars["talent_sums"][talent_name],
                individual_max_level["talent_lvl"],
            ],
        )
        # Store the capped talent sum
        _vars["talent_sums_capped"][talent_name] = var_talent_sum_capped

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
            _vars["talent_sums_capped"][group_talent_name] >= data["talent_lvl"]
        ).OnlyEnforceIf(group_has_enough_levels)
        model.Add(
            _vars["talent_sums_capped"][group_talent_name] < data["talent_lvl"]
        ).OnlyEnforceIf(group_has_enough_levels.Not())
        _vars["group_has_enough_level"][group_talent_name] = group_has_enough_levels

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
        _vars["talent_series_interval"][talent_series_name] = []
        # Iterate over pairs of consecutive levels
        for talent_inferior, talent_superior in pairwise(levels):
            # Create a boolean variable to check if the talent level is more than the inferior level
            var_talent_lvl_is_more_than_inferior = model.NewBoolVar(
                name=f"talent_series_{talent_series_name}_greater_than_{talent_inferior}_{talent_superior}"
            )
            # Add constraints based on the boolean variable
            model.Add(
                talent_inferior <= _vars["talent_sums_capped"][talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_more_than_inferior)
            model.Add(
                talent_inferior > _vars["talent_sums_capped"][talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_more_than_inferior.Not())

            # Create a boolean variable to check if the talent level is less than the superior level
            var_talent_lvl_is_less_than_superior = model.NewBoolVar(
                name=f"talent_series_{talent_series_name}_less_than_{talent_superior}_{talent_superior}"
            )
            # Add constraints based on the boolean variable
            model.Add(
                talent_superior > _vars["talent_sums_capped"][talent_series_name]
            ).only_enforce_if(var_talent_lvl_is_less_than_superior)
            model.Add(
                talent_superior <= _vars["talent_sums_capped"][talent_series_name]
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
            _vars["talent_series_interval"][talent_series_name].append(var)
            model.Add(var == talent_inferior).only_enforce_if(var_talent_lvl_is_between)
            model.Add(var == 0).only_enforce_if(var_talent_lvl_is_between.Not())

    ## Objective
    for name, var in _vars["talent_sums_capped"].items():
        if name not in _vars["talent_sums_final"] and name not in talent_series_names:
            newvar = model.NewIntVar(name=f"talent_sums_final_{name}", lb=0, ub=30)
            _vars["talent_sums_final"][name] = newvar
            if name in group_talent_names:
                model.Add(_vars["talent_sums_final"][name] == var).only_enforce_if(
                    _vars["group_has_enough_level"][name]
                )
                model.Add(_vars["talent_sums_final"][name] == 0).only_enforce_if(
                    _vars["group_has_enough_level"][name].Not()
                )
            else:
                model.Add(_vars["talent_sums_final"][name] == var)
    for name, var_list in _vars["talent_series_interval"].items():
        if name not in _vars["talent_sums_final"]:
            newvar = model.NewIntVar(name=f"talent_sums_final_{name}", lb=0, ub=30)
            _vars["talent_sums_final"][name] = newvar
            model.Add(_vars["talent_sums_final"][name] == sum(var_list))

    # Add additionnal value for free jewels emplacements
    lvl_emplacements = {}
    for lvl in range(1, 4):
        lvl_emplacements[lvl] = []
        for armor_piece, _dict in _vars["jewel_emplacement_sums"].items():
            lvl_emplacements[lvl].append(_dict[f"lvl{lvl}"])

    maximize_nb_of_free_jewels = []
    maximize_nb_of_free_jewels.extend(
        (sum(lvl_emplacements[lvl]) - sum(_vars["jewel_uses_integers"][lvl].values()))
        * 10**3
        * 10**lvl
        for lvl in range(1, 4)
    )
    # Add additional objective value for eventual addional talents (still minimize nb of jewels)
    nb_of_talents = sum(_vars["talent_sums_final"].values())

    # Add optional strict talent optimization
    talent_objective = (
        sum(
            _vars["talent_sums_final"][talent] * 10**talent_weight
            for talent, talent_weight in wanted_talents.items()
        )
        * 10**9
    )
    # Add a penalty if talent lvl is greater than the objective lvl
    abs_diff_vars = []
    for talent_name, talent_var in _vars["talent_sums_final"].items():
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

    ic(f"Solver status: {solver_statuses[status]}")
    solution = {"jewels": {}, "weapon": weapon.to_dicts()[0]["name"]}
    for armor_piece, var_dict in _vars["use_armor_piece_booleans"].items():
        for name, var in var_dict.items():
            if solver.value(var) == 1:
                solution[armor_piece] = name
                # ic(armor.filter(pl.col("name") == name))
    for charm, var in _vars["use_charm_booleans"].items():
        if solver.value(expression=var) == 1:
            solution["charm"] = charm
            # display(charms.filter(pl.col("name") == charm))
    for jewel_lvl, var_dict in _vars["jewel_uses_integers"].items():
        for jewel_name, var in var_dict.items():
            if solver.value(var) > 0:
                solution["jewels"][jewel_name] = solver.value(var)
    ic(_vars["talent_sums_final"]["Tyrannie du Gore Magala"])
    ic(_vars["talent_sums_final"]["Volonté de l'Anjanath tonnerre"])
    ic(model.model_stats())
    return solution


def get_talents_from_solution(solution: dict) -> pl.DataFrame:
    solution_talents = defaultdict(int)
    for jewel_name, quantity in solution["jewels"].items():
        for row in (
            jewels.filter(pl.col("name") == jewel_name)
            .explode("jewel_talent_list")
            .select(
                pl.col("jewel_talent_list").struct.field("name").alias("talent_name"),
                pl.col("jewel_talent_list").struct.field("lvl").alias("talent_lvl"),
            )
            .to_dicts()
        ):
            solution_talents[row["talent_name"]] += row["talent_lvl"]
    for row in (
        weapons.filter(pl.col("name") == solution["weapon"])
        .explode("talents")
        .select(
            pl.col("talents").struct.field("lvl").alias("talent_lvl"),
            pl.col("talents").struct.field("name").alias("talent_name"),
        )
        .to_dicts()
    ):
        solution_talents[row["talent_name"]] += row["talent_lvl"]

    for armor_piece in armor["piece"].unique().sort().to_list():
        if armor_piece not in solution.keys():
            continue
        for row in armor.filter(pl.col("name") == solution[armor_piece]).to_dicts():
            solution_talents[row["talent_name"]] += row["talent_level"]

    for row in charms.filter(pl.col("name") == solution["charm"]).to_dicts():
        solution_talents[row["talent_name"]] += row["talent_lvl"]

    _temp = []
    _temp.extend(
        {
            "talent_name": key,
            "talent_lvl": value,
        }
        for key, value in solution_talents.items()
    )
    return (
        pl.DataFrame(_temp)
        .sort("talent_lvl")
        .join_asof(
            talents.explode("levels")
            .select(
                pl.col("name").alias("talent_name"),
                pl.col("levels").struct.field("lvl").alias("talent_lvl"),
                pl.col("description").alias("general_description"),
                pl.col("levels")
                .struct.field("description")
                .alias("talent_description"),
            )
            .sort("talent_lvl"),
            by="talent_name",
            on="talent_lvl",
        )
        .sort(["talent_lvl", "talent_name"], descending=[True, False])
    )


if __name__ == "__main__":
    # weapon_name = {"name": "Lame d'espoir"}
    weapon_name = weapons.filter(pl.col("name") == "Lame d'espoir").to_dicts()[0]
    talent_list = [
        {"name": "Tyrannie du Gore Magala", "target_level": 2, "weight": 1},
        {"name": "Volonté de l'Anjanath tonnerre", "target_level": 2, "weight": 1},
    ]
    print(solve(weapon_name, talent_list))

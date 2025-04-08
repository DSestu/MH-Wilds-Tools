import polars as pl

armor = pl.read_parquet("data/armor_pieces.parquet")
charms = pl.read_parquet("data/charms.parquet")
jewels = pl.read_parquet("data/jewels.parquet")
talents = pl.read_parquet("data/talents.parquet")
weapons = pl.read_parquet("data/weapons.parquet")


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
            .map_elements(lambda s: sum(s.values()), return_dtype=pl.Int64)
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
    weapon_name = solution["weapon"]
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

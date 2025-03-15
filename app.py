import gradio as gr
import polars as pl

from solver import (
    get_jewel_markdown,
    get_markdown_from_solution,
    get_talents_from_solution,
    solve,
)

quests_parquet_path = "data/quests.parquet"
quests_dataframe = (
    pl.read_parquet(quests_parquet_path)
    .unique()
    .sort("item", "quantity", descending=[False, True])
)

armor = pl.read_parquet("data/armor_pieces.parquet")
charms = pl.read_parquet("data/charms.parquet")
jewels = pl.read_parquet("data/jewels.parquet")
talents = pl.read_parquet("data/talents.parquet")
weapons = pl.read_parquet("data/weapons.parquet")
unique_talents = talents["name"].unique().sort().to_list()


def quest_rewards() -> None:
    gr.Markdown("# Liste des récompenses de quêtes")

    gr_quest_reward_filter_dropdown = gr.Dropdown(
        choices=[(none_value := "0_None")]
        + quests_dataframe["item"].unique().sort().to_list(),
        interactive=True,
        label="Filtre de récompense",
        value=none_value,
    )
    gr_quests_dataframe = gr.DataFrame(quests_dataframe)

    gr_quest_reward_filter_dropdown.change(
        fn=lambda quest_reward_filter: quests_dataframe.filter(
            pl.col("item") == quest_reward_filter
        )
        if quest_reward_filter != none_value
        else quests_dataframe,
        inputs=gr_quest_reward_filter_dropdown,
        outputs=gr_quests_dataframe,
    )


def build_solver() -> None:
    gr.Markdown(open("./optimized_header.md", mode="r", encoding="utf-8").read())
    talents_state = gr.State([])
    solution = gr.State({})
    target_levels = gr.State({})

    with gr.Accordion(
        "Tous les talents: Cliquer sur une ligne rajoute le talent aux souhaits",
        open=True,
    ):
        talent_type_radio = gr.Radio(
            label="Type",
            choices=["Tout", "Equip", "Group", "Series", "Weapon"],
            value="Tout",
        )
        fulltext_search = gr.Text(label="Recherche")
        talent_dataframe_display = gr.DataFrame(
            talents
            #
            .explode("levels")
            .select(
                pl.col("name").alias("Talent"),
                pl.col("levels").struct.field("lvl").alias("Niveau"),
                pl.col("levels").struct.field("description").alias("Détail"),
                pl.col("description").alias("Description"),
            )
            .sort("Talent", "Niveau"),
            interactive=False,
        )

        talent_type_radio.change(
            lambda talent_type, text: talents.filter(pl.col("group") == talent_type)
            .explode("levels")
            .select(
                pl.col("name").alias("Talent"),
                pl.col("levels").struct.field("lvl").alias("Niveau"),
                pl.col("levels").struct.field("description").alias("Détail"),
                pl.col("description").alias("Description"),
            )
            .filter(
                (
                    pl.col("Talent")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Détail")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Description")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
            )
            .sort("Talent", "Niveau")
            if talent_type != "Tout"
            else talents.explode("levels")
            .select(
                pl.col("group").alias("Type"),
                pl.col("name").alias("Talent"),
                pl.col("levels").struct.field("lvl").alias("Niveau"),
                pl.col("levels").struct.field("description").alias("Détail"),
                pl.col("description").alias("Description"),
            )
            .filter(
                (
                    pl.col("Talent")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Détail")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Description")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
            )
            .sort("Type", "Talent", "Niveau"),
            inputs=[talent_type_radio, fulltext_search],
            outputs=talent_dataframe_display,
        )
        fulltext_search.change(
            lambda talent_type, text: talents.filter(pl.col("group") == talent_type)
            .explode("levels")
            .select(
                pl.col("name").alias("Talent"),
                pl.col("levels").struct.field("lvl").alias("Niveau"),
                pl.col("levels").struct.field("description").alias("Détail"),
                pl.col("description").alias("Description"),
            )
            .filter(
                (
                    pl.col("Talent")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Détail")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Description")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
            )
            .sort("Talent", "Niveau")
            if talent_type != "Tout"
            else talents.explode("levels")
            .select(
                pl.col("group").alias("Type"),
                pl.col("name").alias("Talent"),
                pl.col("levels").struct.field("lvl").alias("Niveau"),
                pl.col("levels").struct.field("description").alias("Détail"),
                pl.col("description").alias("Description"),
            )
            .filter(
                (
                    pl.col("Talent")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Détail")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
                | (
                    pl.col("Description")
                    .str.to_lowercase()
                    .str.replace(" ", "")
                    .str.contains(text)
                )
            )
            .sort("Type", "Talent", "Niveau"),
            inputs=[talent_type_radio, fulltext_search],
            outputs=talent_dataframe_display,
        )

    gr.Markdown("# Talents")
    talents_dropdown = gr.Dropdown(
        choices=unique_talents,
        label="Selectionner des talents",
        interactive=True,
        value=None,
    )

    def add_task(tasks, selected_task):
        if selected_task:
            return tasks + [
                {"name": selected_task, "weight": 1, "target_level": -1}
            ], None
        return tasks, None

    # Add a callback to add a talent when clicking on the dataframe
    def add_task_from_dataframe(tasks, df, event: gr.SelectData):
        row = df.iloc[event.index[0]]
        return tasks + [{"name": row["Talent"], "weight": 1, "target_level": -1}]

    talent_dataframe_display.select(
        add_task_from_dataframe,
        inputs=[talents_state, talent_dataframe_display],
        outputs=talents_state,
    )
    talents_dropdown.change(
        add_task, [talents_state, talents_dropdown], [talents_state, talents_dropdown]
    )

    @gr.render(inputs=talents_state)
    def render_talent_list(task_list):
        gr.Markdown(f"### Talents ({len(task_list)})")
        for task in task_list:
            with gr.Row():
                gr.Textbox(
                    task["name"],
                    label="Talent",
                )
                target_level_input = gr.Number(
                    value=task["target_level"],
                    label="Niveau (-1 = max)",
                    interactive=True,
                )
                weight_input = gr.Number(
                    value=task["weight"],
                    label="Priorité (Haute valeur = priorité + haute)",
                    interactive=True,
                )
                target_level_input.change(
                    lambda new_value, tasks, task=task: [
                        t.update({"target_level": new_value}) or t
                        for t in tasks
                        if t is task
                    ][0]
                    and tasks,
                    inputs=[target_level_input, talents_state],
                    outputs=talents_state,
                )

                weight_input.change(
                    lambda new_value, tasks, task=task: [
                        t.update({"weight": new_value}) or t for t in tasks if t is task
                    ][0]
                    and tasks,
                    inputs=[weight_input, talents_state],
                    outputs=talents_state,
                )

                delete_btn = gr.Button("Retirer", scale=0, variant="stop")

                def delete(task=task):
                    task_list.remove(task)
                    return task_list

                delete_btn.click(delete, None, [talents_state])

    gr.Markdown("# Armes")
    default_weapon_type = weapons["class"].unique().sort().to_list()[0]
    default_weapon = (
        weapons.filter(pl.col("class") == default_weapon_type)["name"]
        .unique()
        .sort()
        .to_list()[0]
    )
    default_weapon_data = weapons.filter(pl.col("name") == default_weapon).to_dicts()[0]
    selected_weapon = gr.State(default_weapon_data)
    with gr.Row():
        dropdown_weapon_type = gr.Dropdown(
            choices=(c := weapons["class"].unique().sort().to_list()),
            label="Type d'arme",
            value=c[0],
            interactive=True,
        )
        dropdown_selected_weapon = gr.Dropdown(
            choices=(
                choices := weapons.filter(pl.col("class") == c[0])["name"]
                .unique()
                .sort()
                .to_list()
            ),
            label="Weapons",
            value=choices[0],
        )
        dropdown_weapon_type.change(
            lambda weapon_type: gr.update(
                choices=(
                    c := weapons.filter(pl.col("class") == weapon_type)["name"]
                    .unique()
                    .sort()
                    .to_list()
                ),
                value=c[0],
            ),
            inputs=dropdown_weapon_type,
            outputs=[dropdown_selected_weapon],
        )

    dropdown_selected_weapon.change(
        lambda selected_weapon: weapons.filter(
            pl.col("name") == selected_weapon
        ).to_dicts()[0],
        inputs=dropdown_selected_weapon,
        outputs=selected_weapon,
    )

    solve_button = gr.Button("Optimiser", variant="primary")
    solve_button.click(solve, [selected_weapon, talents_state], solution)

    with gr.Row():
        titles = []
        for armor_piece in ["Tête", "Torse", "Bras", "Taille", "Jambes"]:
            var = gr.State(armor_piece)
            with gr.Column():
                titles.append(gr.Markdown(f"## {armor_piece}"))
                solution.change(
                    lambda x, var_armor_piece: f"## {x[var_armor_piece]}",
                    inputs=[solution, var],
                    outputs=titles[-1],
                )

                var_piece = gr.State()
                solution.change(
                    lambda x, var_armor_piece: armor.filter(
                        pl.col("name") == x[var_armor_piece]
                    ),
                    inputs=[solution, var],
                    outputs=var_piece,
                )
                markdown_talent_list = gr.Markdown()
                var_piece.change(
                    lambda x: "\n\n".join(
                        [
                            row["talent_name"] + " +" + str(row["talent_level"])
                            for row in x.sort(
                                ["talent_level", "talent_name"],
                                descending=[True, False],
                            ).to_dicts()
                        ]
                    ),
                    inputs=var_piece,
                    outputs=markdown_talent_list,
                )

                markdown_jewel_list = gr.Markdown()
                var_piece.change(
                    lambda x: "\n\n".join(
                        [
                            f"- Jewel {i}: "
                            + (
                                str(
                                    x.sort(
                                        ["talent_level", "talent_name"],
                                        descending=[True, False],
                                    ).to_dicts()[0][f"jewel_{i}"]
                                )
                            )
                            for i in range(1, 4)
                            if x.sort(
                                ["talent_level", "talent_name"],
                                descending=[True, False],
                            ).to_dicts()[0][f"jewel_{i}"]
                            != 0
                        ]
                    ),
                    inputs=var_piece,
                    outputs=markdown_jewel_list,
                )
            piece = armor.filter(pl.col("name") == armor_piece)

    gr.Markdown("---")
    with gr.Row():
        solution_jewel_markdown_armor = gr.Markdown()
        solution_jewel_markdown_weapon = gr.Markdown()
    solution_markdown = gr.Markdown()

    solution.change(
        get_jewel_markdown,
        inputs=solution,
        outputs=[solution_jewel_markdown_armor, solution_jewel_markdown_weapon],
    )

    solution.change(
        get_markdown_from_solution, inputs=solution, outputs=solution_markdown
    )


with gr.Blocks() as demo:
    gr.Markdown("# Monster hunter Wilds tools")
    with gr.Tab("Optimization d'équipements"):
        build_solver()
    with gr.Tab("Liste des récompenses de quêtes"):
        quest_rewards()


if __name__ == "__main__":
    demo.launch(
        debug=True,
    )

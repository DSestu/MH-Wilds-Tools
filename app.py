import gradio as gr
import polars as pl

from solver import get_talents_from_solution, solve

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
    talents = gr.State([])
    solution = gr.State({})
    target_levels = gr.State({})
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

    talents_dropdown.change(
        add_task, [talents, talents_dropdown], [talents, talents_dropdown]
    )

    @gr.render(inputs=talents)
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
                    inputs=[target_level_input, talents],
                    outputs=talents,
                )

                weight_input.change(
                    lambda new_value, tasks, task=task: [
                        t.update({"weight": new_value}) or t for t in tasks if t is task
                    ][0]
                    and tasks,
                    inputs=[weight_input, talents],
                    outputs=talents,
                )

                delete_btn = gr.Button("Retirer", scale=0, variant="stop")

                def delete(task=task):
                    task_list.remove(task)
                    return task_list

                delete_btn.click(delete, None, [talents])

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
    with gr.Row():
        with gr.Column():
            weapon_name = gr.Markdown()
            weapon_image = gr.Image(
                None,
                show_label=False,
                container=False,
            )
        weapon_atk = gr.Markdown()

    dropdown_selected_weapon.change(
        lambda selected_weapon: weapons.filter(
            pl.col("name") == selected_weapon
        ).to_dicts()[0],
        inputs=dropdown_selected_weapon,
        outputs=selected_weapon,
    )

    solve_button = gr.Button("Solve")
    solve_button.click(solve, [selected_weapon, talents], solution)

    selected_weapon.change(
        lambda selected_weapon: gr.Image(
            selected_weapon["img"],
            show_label=False,
            container=False,
        ),
        inputs=selected_weapon,
        outputs=weapon_image,
    )
    selected_weapon.change(
        lambda selected_weapon: gr.Markdown(f"# {selected_weapon['name']}"),
        inputs=selected_weapon,
        outputs=weapon_name,
    )

    selected_weapon.change(
        lambda selected_weapon: gr.Markdown(f"## Attaque: {selected_weapon['raw']}"),
        inputs=selected_weapon,
        outputs=weapon_atk,
    )
    talents_display = gr.Markdown()
    talents.change(
        lambda tasks: gr.Markdown(f"### Tasks: {tasks}"),
        inputs=talents,
        outputs=talents_display,
    )
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

    solution_display = gr.Json()
    solution.change(
        lambda solution: gr.Json(solution),
        inputs=solution,
        outputs=solution_display,
    )
    solution_talents = gr.DataFrame()
    solution.change(
        lambda solution: get_talents_from_solution(solution),
        inputs=solution,
        outputs=solution_talents,
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

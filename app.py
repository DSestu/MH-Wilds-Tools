import polars as pl
import gradio as gr


quests_parquet_path = "data/quests.parquet"
quests_dataframe = (
    pl.read_parquet(quests_parquet_path)
    .unique()
    .sort("item", "quantity", descending=[False, True])
)


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


with gr.Blocks() as demo:
    gr.Markdown("# Monster hunter Wilds tools")
    with gr.Tab("Liste des récompenses de quêtes"):
        quest_rewards()


if __name__ == "__main__":
    demo.launch(
        debug=True,
    )

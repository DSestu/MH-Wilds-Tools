import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.styles import Style
from rich import traceback
from rich.console import Console
from rich.markdown import Markdown

from .scraper import (
    get_all_charms,
    get_all_jewels,
    get_all_quest_details,
    get_all_skills,
    scrape_all_armors,
)

console = Console()
traceback.install()

help_str = """
# **Available commands:**

## **scrape**:

- all     : Scrape all game data (quests, armors, charms, jewels, skills)
- quests  : Scrape only quest data
- armors  : Scrape only armor data
- charms  : Scrape only charm data
- jewels  : Scrape only jewel data
- skills  : Scrape only skill data

## **app**:
- start   : Start the application
- stop    : Stop the application

## **Other**:
- exit    : Exit the application
- help    : Display this help message
"""

COMMANDS_TREE = {
    "help": None,
    "exit": None,
    "scrape": {
        "all": None,
        "armors": None,
        "charms": None,
        "jewels": None,
        "quests": None,
        "skills": None,
    },
    "app": {
        "start": None,
        "stop": None,
    },
}

COMMAND_STYLE = Style.from_dict(
    {
        "prompt": "#00aa00 bold",  # Green prompt
        "completion-menu.completion": "bg:#008888 #ffffff",  # Cyan background with white text
        "completion-menu.completion.current": "bg:#00aaaa #000000",  # Light cyan with black text
        "scrollbar.background": "bg:#88aaaa",  # Light cyan scrollbar
        "scrollbar.button": "bg:#222222",  # Dark gray scrollbar button
    }
)


async def command_dispatcher(command: str) -> None:
    # Transform multiple spaces into one
    command = " ".join(command.split())

    command_parts = command.split()
    main_command = command_parts[0]
    subcommand = command_parts[1] if len(command_parts) > 1 else None

    match main_command:
        case "help":
            console.print(Markdown(help_str))
            return
        case "scrape":
            match subcommand:
                case "all":
                    print("Scraping all data...")
                    print("Scraping quest details...")
                    await get_all_quest_details()
                    print("Scraping armor data...")
                    await scrape_all_armors()
                    print("Scraping charms data...")
                    await get_all_charms()
                    print("Scraping jewels data...")
                    await get_all_jewels()
                    print("Scraping skills data...")
                    await get_all_skills()
                case "quests":
                    print("Scraping quests data...")
                    await get_all_quest_details()
                case "armors":
                    print("Scraping armor data ...")
                    await scrape_all_armors()
                case "charms":
                    print("Scraping charms data ...")
                    await get_all_charms()
                case "jewels":
                    print("Scraping jewels data ...")
                    await get_all_jewels()
                case "skills":
                    print("Scraping skills data ...")
                    await get_all_skills()
                case _:
                    print("Usage: scrape all")
            return
        case "app":
            if subcommand == "start":
                print("Starting app...")
            elif subcommand == "stop":
                print("Stopping app...")
            else:
                print("Usage: app [start|stop]")
            return
        case _:
            print("Invalid command. Type 'help' for a list of commands.")
            return


async def main():
    command_completer = NestedCompleter.from_nested_dict(COMMANDS_TREE)
    session = PromptSession(
        completer=command_completer,
        complete_while_typing=True,
        enable_history_search=True,
    )

    console.print(
        Markdown(f"""
# Welcome to MH Wilds Tools scraper CLI!

> The goal of this CLI is to provide a way to scrape data from Kiranico in order to update the app database.

{help_str}

Type any command to begin!
    """)
    )
    while True:
        try:
            text = await session.prompt_async(
                "> ",
            )
            command = text.strip().lower()
            match command:
                case "exit":
                    print("Goodbye!")
                    break
                case _:
                    await command_dispatcher(command=command)
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nUse 'exit' to quit")
        except EOFError:
            # Handle Ctrl+D
            print("\nGoodbye!")
            break


def entrypoint() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

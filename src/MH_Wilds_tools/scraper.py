import asyncio
import contextlib
import itertools
import os
import pathlib
from typing import Callable

import polars as pl
from pydoll.browser.chrome import Chrome
from pydoll.browser.options import Options
from pydoll.browser.page import Page
from pydoll.constants import By
from pydoll.element import WebElement
from tqdm import tqdm

SCRAPE_CHUNK_PAGES = 10
HEADLESS = False
ROOT_URL = "https://mhwilds.kiranico.com"

repo_path = pathlib.Path(__file__).parent.parent.parent


async def highlight(self, element: WebElement, time: int = 5) -> None:
    """
    Temporarily highlight a web element by adding a red border and then restoring its original style.

    Args:
        element (WebElement): The web element to highlight.
        time (int, optional): Duration of the highlight in seconds. Defaults to 5.

    Briefly draws attention to a specific web element by adding a red border with a smooth transition,
    then restores the element's original styling after a specified time interval.
    """
    original_style = element.get_attribute("style")
    await self.execute_script(
        """
    argument.style.border = '3px solid red';
    argument.style.transition = 'border 0.3s ease-in-out';
    """,
        element,
    )
    await asyncio.sleep(time)

    await self.execute_script(
        f"""
        argument.setAttribute('style', '{original_style}');
        """,
        element,
    )


async def parallel_scrap(
    fn: Callable,
    chunk_iterator: list,
    chunk_key: str,
    max_concurrent: int = 60,
    **kwargs,
) -> list:
    """
    Asynchronously scrape data with a maximum number of concurrent tasks.

    Args:
        fn (Callable): The async function to call for each element in the iterator.
        chunk_iterator (list): The list of elements to be processed.
        chunk_key (str): The key name to pass each element to the function.
        max_concurrent (int, optional): Maximum number of concurrent tasks. Defaults to 60.
        **kwargs: Additional keyword arguments to pass to the scraping function.

    Returns:
        list: Aggregated results from all processed elements.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    all_data = []
    tasks = []

    async def process_item(element):
        async with semaphore:
            return await fn(**{chunk_key: element}, **kwargs)

    # Create all tasks
    for element in chunk_iterator:
        tasks.append(process_item(element))

    # Process tasks with progress bar
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await task
        all_data.append(result)

    return all_data


def get_options(
    headless: bool = False,
    chrome_user_data: str = os.path.join(os.getcwd(), "chrome_user_data"),
) -> Options:
    """
    Configure and return Chrome WebDriver options for web automation.

    Args:
        headless (bool, optional): Whether to run Chrome in headless mode. Defaults to False.
        chrome_user_data (str, optional): Path to Chrome user data directory.
            Defaults to a 'chrome_user_data' directory in the current working directory.

    Returns:
        Options: Configured Chrome WebDriver options with specific settings for web scraping.
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-dev-shm-usage")

    options.add_argument(f"--user-data-dir={chrome_user_data}")
    options.add_argument("--profile-directory=Default")
    return options


Page.highlight = highlight


async def get_quest_rewards(
    quest: dict[str, str], browser, root_url: str
) -> list[dict[str, str]]:
    href = quest["href"]
    page = await browser.get_page()
    await page.go_to(url=f"{root_url}{href}")

    recompenses_element = await page.find_element(
        By.XPATH,
        "//h3[contains(text(), 'Récompenses')]",
    )
    parent_element = await recompenses_element.find_element(By.XPATH, "..")

    recompenses_list = []

    recompenses = await parent_element.find_elements(By.TAG_NAME, "tr")
    for recompense in recompenses:
        value = await recompense.find_elements(By.TAG_NAME, "td")
        value = value[0]
        item = await value.find_element(By.TAG_NAME, "a")
        item = await item.get_element_text()
        quantity = await value.get_element_text()
        quantity = quantity.replace(item, "").strip().replace("x", "")
        match quantity:
            case "":
                quantity = 1
            case _:
                quantity = int(quantity)
        recompenses_list.append({"item": item, "quantity": quantity})
    await page.close()
    return {
        "name": quest["name"],
        "href": quest["href"],
        "rewards": recompenses_list,
    }


async def get_quest_details(quest: WebElement) -> dict[str, str]:
    quest_name_element = await quest.find_elements(By.TAG_NAME, "td")
    quest_name_element = quest_name_element[0]
    quest_name = await quest_name_element.find_element(By.TAG_NAME, "a")
    return {
        "name": await quest_name.get_element_text(),
        "href": quest_name.get_attribute("href"),
    }


async def get_all_quest_details() -> None:
    """
    Scrapes quest details and rewards from a web page using an asynchronous browser automation approach.

    This code block performs the following key operations:
    - Launches a Chrome browser with predefined options
    - Navigates to a quest page
    - Extracts quest details and rewards
    - Processes the collected data into a structured DataFrame
    - Saves the quest data to a Parquet file

    The scraping process involves:
    - Gathering quest details using asyncio
    - Fetching rewards in chunks to manage memory and performance
    - Transforming the collected data using Polars and Pandas
    """
    quest_page = f"{ROOT_URL}/fr/data/missions"
    async with Chrome(options=get_options(headless=HEADLESS)) as browser:
        await browser.start()
        page = await browser.get_page()
        await page.go_to(url=quest_page, timeout=10)

        quest_scroll_element = await page.find_element(
            By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div[2]"
        )
        quests = await quest_scroll_element.find_elements(By.TAG_NAME, "tr")

        all_quests = await asyncio.gather(
            *[get_quest_details(quest) for quest in quests]
        )

        all_quests_rewards = await parallel_scrap(
            get_quest_rewards,
            chunk_key="quest",
            chunk_iterator=all_quests,
            browser=browser,
            root_url=ROOT_URL,
            max_concurrent=SCRAPE_CHUNK_PAGES,
        )

        quest_df = (
            pl.DataFrame(all_quests_rewards)
            #
            .explode("rewards")
            .with_columns(
                pl.col("rewards").struct.field("item").alias("item"),
                pl.col("rewards").struct.field("quantity").alias("quantity"),
            )
            .drop("rewards")
            .filter(pl.col("item").is_not_null())
            .sort(pl.all())
            .to_pandas()
            .get(["name", "item", "quantity"])
        )
        quest_df.to_parquet(os.path.join(repo_path, "data", "quests.parquet"))


async def extract_armor_data(
    browser,
    root_url: str,
    href: str,
) -> list[dict[str, str]]:
    """
    Extract armor data from a web page, parsing talent table rows to collect piece details.

    Navigates to a specific URL, finds the talent table, and extracts information about
    each armor piece including its name, jewel levels, and associated talents.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing armor piece details with
        keys 'piece', 'name', 'jewels', and 'talents'.
    """
    page = await browser.get_page()
    await page.go_to(url=f"{root_url}{href}")

    # Get skill table
    talent_table = await page.find_element(
        By.XPATH,
        '//th[contains(text(), "Talents de l\'équipement")]',
    )
    talent_table = await talent_table.find_element(By.XPATH, "..")
    talent_table = await talent_table.find_element(By.XPATH, "..")

    all_pieces = []
    talent_table_rows = (await talent_table.find_elements(By.TAG_NAME, "tr"))[1:]
    for talent_table_row in talent_table_rows:
        piece, name, jewels, talent = await talent_table_row.find_elements(
            By.TAG_NAME, "td"
        )
        piece, name, jewels = await asyncio.gather(
            *[x.get_element_text() for x in (piece, name, jewels)]
        )

        # Extract jewel levels
        all_jewels = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
        for char in jewels.replace("[", "").replace("]", ""):
            all_jewels[char] += 1

        # Extract talent information
        all_talents = []
        with contextlib.suppress(Exception):
            talents = await talent.find_elements(By.TAG_NAME, "a")

        for _talent in talents:
            talent_name = await _talent.get_element_text()
            talent_level = int(talent_name.split("+")[-1])
            talent_name = talent_name.split("+")[:-1]
            talent_name = "+".join(talent_name).strip()
            all_talents.append(
                {
                    "talent_name": talent_name,
                    "talent_level": talent_level,
                }
            )
            piece_dict = {
                "piece": piece,
                "name": name,
                "jewels": all_jewels,
                "talents": all_talents,
            }
            all_pieces.append(piece_dict)
    await page.close()
    return all_pieces


async def scrape_all_armors() -> None:
    armor_page = f"{ROOT_URL}/fr/data/armor-series"

    async with Chrome(options=get_options(headless=HEADLESS)) as browser:
        await browser.start()
        page = await browser.get_page()
        await page.go_to(armor_page)

        scroll_element = await page.find_element(By.TAG_NAME, "table")
        hrefs = await scroll_element.find_elements(By.TAG_NAME, "a")
        hrefs = [element.get_attribute("href") for element in hrefs]

        all_armor_data = await parallel_scrap(
            fn=extract_armor_data,
            chunk_key="href",
            chunk_iterator=hrefs,
            browser=browser,
            root_url=ROOT_URL,
            max_concurrent=SCRAPE_CHUNK_PAGES,
        )
        all_armor_data = list(itertools.chain.from_iterable(all_armor_data))

    armor_pieces = (
        pl.DataFrame(all_armor_data)
        #
        .explode("talents")
        .with_columns(
            pl.col("talents").struct.field("talent_name").alias("talent_name"),
            pl.col("talents").struct.field("talent_level").alias("talent_level"),
        )
        .drop("talents")
        .with_columns(
            *[
                pl.col("jewels").struct.field(jwl_lvl).alias(f"jewel_{jwl_lvl}")
                for jwl_lvl in ["0", "1", "2", "3", "4"]
            ]
        )
        .drop("jewels")
        .unique()
        .sort(pl.all())
    )
    armor_pieces.write_parquet(os.path.join(repo_path, "data", "armor_pieces.parquet"))


async def extract_charm_row_data(charm_element: WebElement) -> dict[str, str]:
    charm_name = await charm_element.find_element(By.TAG_NAME, "a")
    charm_name = await charm_name.get_element_text()
    charm_href = (await charm_element.find_element(By.TAG_NAME, "a")).get_attribute(
        "href"
    )
    return {"name": charm_name, "href": charm_href}


async def extract_charm_data(
    browser,
    root_url: str,
    charm_element: dict[str, str],
) -> dict[str, str]:
    page = await browser.get_page()
    await page.go_to(f"{root_url}{charm_element['href']}")

    talent_table = await page.find_element(By.TAG_NAME, "tbody")
    talents = await talent_table.find_elements(By.TAG_NAME, "tr")

    charm_talents = []
    for talent in talents:
        name, lvl, desc = await talent.find_elements(By.TAG_NAME, "td")
        name = await name.get_element_text()
        lvl = await lvl.get_element_text()
        desc = await desc.get_element_text()

        lvl = int(lvl.replace("Lv", "").strip())
        charm_talents.append({"name": name, "lvl": lvl})
    charm_element["talents"] = charm_talents
    await page.close()
    return charm_element


async def get_all_charms() -> None:
    async with Chrome(options=get_options(headless=HEADLESS)) as browser:
        charms_url = f"{ROOT_URL}/fr/data/charms"

        await browser.start()
        page = await browser.get_page()
        await page.go_to(url=charms_url)

        scroll_element = await page.find_element(By.TAG_NAME, "table")
        charm_elements = await scroll_element.find_elements(By.TAG_NAME, "tr")

        all_charms = await asyncio.gather(
            *[extract_charm_row_data(charm_element) for charm_element in charm_elements]
        )

        all_charm_data = await parallel_scrap(
            extract_charm_data,
            browser=browser,
            root_url=ROOT_URL,
            chunk_iterator=all_charms,
            chunk_key="charm_element",
            max_concurrent=SCRAPE_CHUNK_PAGES,
        )

    charms_data = (
        pl.DataFrame(all_charm_data)
        #
        .explode("talents")
        .with_columns(
            pl.col("talents").struct.field("name").alias("talent_name"),
            pl.col("talents").struct.field("lvl").alias("talent_lvl"),
        )
        .drop("talents")
        .sort(pl.all())
    )
    charms_data.write_parquet(os.path.join(repo_path, "data", "charms.parquet"))


async def get_jewel_data(browser, url: str) -> dict[str, str | int | list]:
    page = await browser.get_page()
    await page.go_to(url=f"{ROOT_URL}{url}")

    jewel_name = await (await page.find_element(By.TAG_NAME, "h2")).get_element_text()
    jewel_lvl = int(jewel_name.split("[")[-1][0])

    skill_table = await page.find_element(By.TAG_NAME, "table")
    skill_rows = await skill_table.find_elements(By.TAG_NAME, "tr")

    skills = []

    for skill_row in skill_rows:
        skill_name, skill_lvl, skill_description = await skill_row.find_elements(
            By.TAG_NAME, "td"
        )
        href = (await skill_name.find_element(By.TAG_NAME, "a")).get_attribute("href")
        skill_name = (await skill_name.get_element_text()).strip()

        skill_lvl = int((await skill_lvl.get_element_text()).replace("Lv", "").strip())
        skill_description = (await skill_description.get_element_text()).strip()
        skills.append(
            {
                "name": skill_name,
                "lvl": skill_lvl,
                "description": skill_description,
                "href": href,
            }
        )
    await page.close()
    return {
        "name": jewel_name,
        "jewel_lvl": jewel_lvl,
        "jewel_talent_list": skills,
    }


async def get_all_jewels() -> None:
    async with Chrome(options=get_options(headless=HEADLESS)) as browser:
        jewels_url = f"{ROOT_URL}/fr/data/decorations"

        await browser.start()
        page = await browser.get_page()
        await page.go_to(url=jewels_url)

        scroll_element = await page.find_element(By.TAG_NAME, "table")
        jewel_rows = await scroll_element.find_elements(By.TAG_NAME, "tr")

        all_urls = [
            element.get_attribute("href")
            for element in await asyncio.gather(
                *[jewel_row.find_element(By.TAG_NAME, "a") for jewel_row in jewel_rows]
            )
        ]
        all_jewel_data = await parallel_scrap(
            get_jewel_data,
            chunk_iterator=all_urls,
            chunk_key="url",
            browser=browser,
            max_concurrent=SCRAPE_CHUNK_PAGES,
        )

        (
            pl.DataFrame(all_jewel_data).write_parquet(
                os.path.join(repo_path, "data", "jewels.parquet")
            )
        )


async def extract_skill_data(browser, skill: dict) -> dict:
    page = await browser.get_page()
    await page.go_to(skill["href"])

    skill_table = await page.find_element(By.CSS_SELECTOR, ".my-8 tbody")

    all_levels = []
    rows = await skill_table.find_elements(By.TAG_NAME, "tr")
    for row in rows:
        lvl, _, description = await row.find_elements(By.TAG_NAME, "td")

        lvl = int((await lvl.get_element_text()).replace("Lv", ""))
        description = (await description.get_element_text()).strip()

        all_levels.append(
            {
                "lvl": lvl,
                "description": description,
            }
        )
    await page.close()
    return {
        "group": skill["group"],
        "name": skill["name"],
        "description": skill["description"],
        "href": skill["href"],
        "levels": all_levels,
    }


async def get_all_skills() -> None:
    async with Chrome(options=get_options(headless=HEADLESS)) as browser:
        skills_href = f"{ROOT_URL}/fr/data/skills"
        await browser.start()
        page = await browser.get_page()

        await page.go_to(skills_href)

        all_skills = []

        scroll_groups = []
        for group_name in ["Weapon", "Equip", "Group", "Series"]:
            scroll_groups.append(
                await page.find_element(
                    By.XPATH, f"//h3[contains(text(), '{group_name}')]/.."
                )
            )

        for scroll_group in scroll_groups:
            group_name = await (
                await scroll_group.find_element(By.TAG_NAME, "h3")
            ).get_element_text()

            skill_elements = await scroll_group.find_elements(By.TAG_NAME, "tr")
            for skill_element in skill_elements:
                name, description = await skill_element.find_elements(By.TAG_NAME, "td")

                href = ROOT_URL + (
                    await name.find_element(By.TAG_NAME, "a")
                ).get_attribute("href")

                name = (await name.get_element_text()).strip()
                description = (
                    (await description.get_element_text()).strip().replace("\n", " ")
                )

                all_skills.append(
                    {
                        "group": group_name,
                        "name": name,
                        "description": description,
                        "href": href,
                    }
                )
        all_skill_data = await parallel_scrap(
            extract_skill_data,
            chunk_iterator=all_skills,
            chunk_key="skill",
            browser=browser,
            max_concurrent=SCRAPE_CHUNK_PAGES,
        )
        (
            pl.DataFrame(all_skill_data)
            #
            .write_parquet(os.path.join(repo_path, "data", "talents.parquet"))
        )


# if __name__ == "__main__":
#     asyncio.run(get_all_quest_details())

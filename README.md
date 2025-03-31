---
title: MH Wilds Armor Optimizer
emoji: 🎮
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
short_description: Armor optimization tool for Monster Hunter Wilds
---

# Monster Hunter Wilds Armor Optimizer 🛡️

## Overview

A precise armor optimization tool for Monster Hunter Wilds that calculates the most efficient equipment combinations based on your desired skills.

## Core Features 🔧

- Skill selection and prioritization
- Equipment slot optimization
- Weapon-specific calculations
- Charm and decoration integration
- Complete loadourt solutions

## Setup

You will need `uv` to be installed.

Check here [the installation instructions.](https://docs.astral.sh/uv/getting-started/installation/)

## Run

```{bash}
uv run gradio app.py
```

Note: The `requirements.txt` file is just for Hugging face space setup.

## How It Works ⚙️

1. Select your desired skills
2. Set skill priorities (1-10)
3. Choose your weapon
4. Get optimal armor combinations

## Technical Details 💻

Uses Google OR-Tools for:

- Equipment combination calculations
- Skill level optimization
- Slot efficiency maximization
- Special skill interaction handling

## Usage Guide 📖

1. Select skills from database
2. Set priorities and targets
3. Choose weapon type
4. Run optimization
5. View recommended loadout

## Output Format 📋

Results include:

- Full armor set (Head/Chest/Arms/Waist/Legs)
- Recommended charm
- Decoration placement
- Achieved skill levels

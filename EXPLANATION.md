> [**Note: If you are on Github, visit this page instead to have proper display of images**](https://huggingface.co/spaces/Nnugget/MH_Wilds_tools/blob/main/EXPLANATION.md)

# Building a multi-objective gear optimization system for a popular video game using Constraint Programming - Monster Hunter Wilds

In this article, we'll dive deep into the core mechanics of Monster Hunter's equipment and skill system. Understanding these fundamentals is crucial before we explore how to optimize and build the perfect gear sets. We'll examine how armor pieces, skills, and their synergies work together to create unique hunting experiences.

To tackle the complex challenge of finding optimal gear combinations, we'll leverage the power of constraint programming (CP). We'll progressively explore this optimization technique through multiple difficulty levels - starting with basic armor selection, then incorporating skills, and finally addressing advanced constraints like decoration slots etc. This mathematical approach will help us discover the most efficient gear combinations while satisfying our desired skill requirements.

While we use a video game as an example here, the constraint programming approach we'll explore can be applied to many real-world optimization problems. Some practical applications include:

- Supply chain optimization: selecting the best combination of warehouses and transportation routes
- Manufacturing: optimizing production schedules and resource allocation
- Staff scheduling: creating optimal shift patterns while satisfying various constraints
- Portfolio optimization: selecting the best mix of investments under budget constraints
- Facility location: determining optimal locations for new facilities while considering multiple factors
- Vehicle routing: optimizing delivery routes for logistics companies
- Product mix optimization: deciding the best combination of products to manufacture given resource constraints

These real-world problems share similar characteristics with our Monster Hunter example - they involve selecting from multiple options while satisfying various constraints to achieve optimal results.

# Explaining the business mechanics prior to optimization

Monster Hunter is a video game franchise by Capcom.

This franchise consists of hunting monsters, giving players monster parts that allow them to create corresponding weapons and armor.

A player (or hunter), has equipment, gear, that they will use to better hunt monsters, and that will heavily affect the gameplay.

A gear is made of a set of armor, a talisman, and a weapon.

The armor is decomposed into multiple parts:

- Head
- Chest
- Arms
- Waist
- Legs

Each of those parts can be equipped independently. This means that we can equip a head of monster A and a torso of monster B.

A player can equip at maximum one head at a time, one torso at a time etc. Though, they can switch whenever they want to modify their gameplay.

### Skills

> Why do different armor pieces lead to different gameplay styles?

This is because there are "skills" in Monster Hunter.

Those skills will add something to the gameplay, a permanent effect to the player as long as they have the corresponding skill in their equipment.

Those may, or may not, have multiple levels.

In the following image, you can see the "Recovery up" skill, which increases the amount of health restored to the player *(for example when eating a healing potion)*. This skill has 3 levels, increasing the amount of health restored for each skill level.

![alt text](./doc/recovery_up.png)

> The core idea of Monster Hunter is to build multiple custom sets of gear that will affect the gameplay due to the multiple skills inherited from the gear. Those skills are at best, working in synergy to propose a strong gameplay variation.

An example of the synergy between skills:

![alt text](./doc/wide_range.png)
![alt text](./doc/free_meal.png)
![alt text](./doc/speed_eating.png)

With those skills, a hunter can eat consumables fast, with a chance to consume it for free, and share consumable effects with their allies. This is a core synergy for supporting teammates during hunts.

In reality, a hunter can accumulate many more skills with an armor set, multiplying the number of different gameplay variations.

### How skill points are acquired?

Skill points are acquired in multiple ways:

- Each individual armor piece has some skill points

- A talisman is a piece of gear that is solely granting specific skill points

- Each weapon has also some skill points

> Additionally, skill points are acquired via jewels

### Jewels

Each armor piece, each weapon, has jewel slots.

There are 3 sizes of jewel slots: 1, 2, and 3.

In the following image, you can see the Conga helm, the Conga mail, and Conga vambraces.

In this example, you can see that the Conga helm has 1 jewel-2 slot. The mail has 3 jewel-1 slots, and the vambraces have 1 jewel-1 slot.

![alt text](./doc/conga_armor.png)

During the game, a hunter will acquire jewels (1, 2 and 3). These jewels can be placed in empty jewel slots.

Each jewel will give skill points, one or multiple, for one or multiple skills. A different jewel will give different skill points.

Additionally, there are some rules:

- A jewel of size 3 can only fit in a jewel slot of size 3
- A jewel of size 2 can fit in a jewel slot of size 2 or 3
- A jewel of size 1 can fit in a jewel slot of size 1, 2 or 3

Only one jewel can be placed in a jewel slot at a time. This means that we can place only one jewel-1 in a jewel-2 or jewel-3 emplacement. Emplacement sizes can be considered as "tiers".

### Armor skills & weapon skills

In the latest Monster Hunter game, Monster Hunter Wilds, skills are either armor skills or weapon skills.

Armor skills will be found only on armor pieces. Additionally, armor jewel slots can only take armor jewel types.

Weapon skills *(which are more offensive)* will be found only on weapons. Weapon jewel slots will only take weapon jewel types.

This means that there are 2 big jewel categories: armor type and weapon type, which can only be placed in the corresponding gear slots.

### Group skills

The skills shown before are "standard" skills *(armor skills in the examples)*.

Those skills give a benefit to the player as soon as there is at least one skill point equipped.

However, there is also another kind of skills: group skills.

You can see an example below.

![alt text](./doc/leathercraft.png)

Group skills are special because only one skill point does nothing.

Group skills are activated only if there are at least X skill points. In the example above, at least 3 skill points.

In the game, each armor piece will give at maximum one group skill. Also, group skills can't be obtained via jewels.

This means that group skills are obtained via careful composition of armor pieces.

### Series/set bonus skills

Series skills are very similar to group skills, however they have multiple activation thresholds.

In the example below, you can see that the set bonus has 2-pieces and 4-pieces activation thresholds.

![alt text](./doc/nu_udra_mutiny.png)

If the hunter has:

- 1 skill point: no bonus
- 2 skill points: 2-pieces bonus
- 3 skill points: 2-pieces bonus
- 4 skill points: 4-pieces bonus
- 5 skill points: 4-pieces bonus

### Conclusion on game mechanics

A hunter will build custom armor sets that will affect their gameplay.

They will need to carefully assemble 5 different armor pieces, a charm, a weapon, and different types of jewels.

The hunter has to take care of group skills, series skills.

They also have to be very careful about the jewel slots that the gear provides.

This is a pure operational research task, implemented as a game.

This can be done by a human, but optimal solutions are very hard to find, especially as the game contents expand through numerous updates.

This is why we will drill into operational research, via Constraint Programming (CP).

# Using Constraint Programming for Monster Hunter Wilds gear optimization

## Setup

For this project, we are simply going to use:

- Google OR-Tools for Constraint Programming
- Polars for dataframe manipulation
- Pandas for some `.to_markdown()` prints
- Arrow for Polars -> Pandas transformations

Data collection on which the multiple datasets are based on won't be covered here, as we primarily focus on the CP part. However, those were scraped from Kiranico (in French) using Selenium.

## What is Constraint Programming?

Constraint Programming (CP) is a mathematical optimization technique used to solve complex problems by specifying constraints that must be satisfied. It's particularly useful for problems involving discrete choices, like selecting armor pieces in Monster Hunter Wilds.

### Key characteristics of CP

1. Variables can be:
   - Integer (whole numbers)
   - Binary (0 or 1)

2. The problem is defined by:
   - An objective function to maximize or minimize
   - A set of constraints that must be satisfied
   - Domains for variables

### Why is CP original and useful?

1. **Declarative approach**
   - Instead of writing step-by-step instructions
   - You describe WHAT you want, not HOW to get it
   - The solver figures out the solution path

2. **Guaranteed feasibility**
   - When a solution is found, it satisfies all constraints
   - No need to wonder if constraints are violated

3. **Complex constraints**
   - Can handle hundreds or thousands of interrelated constraints
   - Perfect for complex systems like Monster Hunter's gear mechanics
   - Naturally models "if-then" relationships

4. **Performance**
   - Modern CP solvers are highly optimized
   - Can solve problems with large numbers of constraints
   - Much faster than brute force approaches

## How does CP solve Monster Hunter Wilds gear optimization?

The power of CP lies in its declarative approach to problem-solving. Instead of manually testing countless gear combinations, we:

1. Define the problem mathematically through:
   - Variables representing armor pieces, weapons, and charms
   - Constraints that model Wilds' equipment mechanics and skill system

2. Let specialized solvers find the optimal loadout

The key challenge is accurately modeling Monster Hunter Wilds' intricate equipment system as a constraint satisfaction problem. This includes:

- Representing each piece of gear and their unique properties
- Encoding both group skills and series skills requirements
- Modeling jewel slot configurations and equipment restrictions
- Handling charm and weapon selection constraints

In the following sections, we'll build this model step by step, starting with basic examples and progressively adding complexity to capture the full depth of Monster Hunter Wilds' gear mechanics.

## Basics of CP

In this very basic example, I will show you how to model a very simple problem: finding the maximum of a constrained integer variable.

First we have to define a constraint programming model:

![Basic model instantiation](./doc/code/basic_instantiate_model.png)

Then, we define an integer variable that has an upper bound and a lower bound. We will give an arbitrary high upper bound of 100.

![Define variable with bounds](./doc/code/basic_define_my_var.png)

Here, we will add a constraint that will force the variable to be less than or equal to 50.

![Add constraint](./doc/code/basic_constraint.png)

Finally, we will define our objective function. Here we want to find the solution that maximizes `my_var` value, while fulfilling model constraints.

![Define objective function](./doc/code/basic_objective.png)

Finally, we will instantiate a solver, solve the model and print the solution.

![Solve and print solution](./doc/code/basic_solution.png)

This gives the following output:

```
Solver solution: OPTIMAL
Value of my_var: 50
```

As a conclusion, we:

- Created a constraint programming model
- Defined an integer variable that can go up to 100
- Defined a constraint to tell the model that it can't go more than 50
- Defined an objective function to maximize this variable
- Printed the expected solution of the variable being optimal at a value of 50

Now that we saw the very basics of model instantiation and optimization, we are going to move on with a more applied problem.## Modeling the fact that we can only equip one charm at a time.

To model the single charm equipment restriction:

1. We create a boolean variable for each charm that indicates whether it is equipped (1) or not (0).

2. We add a constraint that ensures the sum of all these boolean variables must be less than or equal to 1.

This mathematically enforces that at most one charm can be equipped at any time, since:

- Each charm's equipped status is represented by 0 (unequipped) or 1 (equipped)
- The sum being ≤ 1 means only one charm can have a value of 1
- If no charms are equipped, the sum will be 0

For example, with 3 charms:

- Valid: [0,0,0] - No charms equipped
- Valid: [1,0,0] - First charm equipped
- Invalid: [1,1,0] - Cannot equip multiple charms

![Basic charm equipped constraint](./doc/code/basic_charm_equipped.png)

This produces the following output:

```raw
Charm Talisman anti-immobilization is equipped
```

### Modeling the fact that a charm is equipped, and adding its associated skills

To model how charms provide skill points, we'll use conditional constraints. Here's how it works:

1. For each charm, we create:
   - A boolean variable indicating if the charm is equipped (true/false)
   - An integer variable representing the skill points provided by this charm

2. We then add two conditional constraints:
   - When the charm is equipped (boolean = true):
     The skill points variable equals the charm's actual skill point value
   - When the charm is not equipped (boolean = false):
     The skill points variable equals 0

This approach ensures that skill points are only counted when a charm is actually equipped, and are set to zero otherwise.

We sum the skill points in this example because we want to:

- Track the total skill points across all equipped charms
- Ensure only equipped charms contribute to the total
- Make it easy to maximize total skill points in the objective function
- Enable constraints based on total skill point thresholds

![Full charm solution implementation](./doc/code/full_charm_solution.png)

The output is the following:

```
Charm: Talisman de botanique IV
Skill: Botaniste
Skill level: 4
```

This same approach of limiting equipped pieces to 1 and tracking skills in a registry can be applied to all armor types (head, chest, arms, waist, legs).

You can see the implementation in the code below:

![Register gear implementation](./doc/code/register_gear.png)

### Capping the maximum level of skills

In Monster Hunter, each skill has a maximum level that can range from 1 to 5. For example, some skills max out at level 3, while others can go up to level 5.

When optimizing armor builds, it's crucial to respect these maximum skill levels. There's no benefit in accumulating more skill points than the maximum level allows. For instance, if a skill has a maximum level of 3, having 5 skill points would waste 2 points that could be better used elsewhere.

To implement this constraint in our optimization system:

1. First, we track the total number of skill points obtained from all equipment pieces
2. Then, we create "capped" variables that limit these totals to each skill's maximum level
3. Finally, we use these capped variables in our optimization objective

This ensures the solver won't waste resources trying to exceed a skill's maximum level, since any additional points beyond the cap won't increase the capped value.
In order to do this, we add this part in the code:

![Skill level capping implementation](./doc/code/capping_skill_levels.png)

Then, we must use those new proxy variables in the objective function, so

```python
model.maximize(obj=var_registry["skill_sum"]["Botaniste"])
```

becomes this

```python
model.maximize(obj=var_registry["skill_sum_capped"]["Botaniste"])
```

### Group skills

Group skills are special skills in Monster Hunter that have unique activation requirements:

- They only activate when reaching a minimum threshold of skill points
- Below the threshold, they provide no benefit (0 points)
- Once activated, they work at full strength
- Additional points beyond the threshold don't increase effectiveness
- Example: A group skill requiring 3 points will:
  - Give 0 points with 1-2 skill points
  - Fully activate at 3+ points

Group skills require a minimum number of skill points to activate. We handle this with two key rules:

1. If total skill points are below the activation threshold:
    - Set the skill sum to 0 since the skill is not active
    - Example: If a group skill needs 3 points to activate but only has 2, it provides no benefit

2. If total skill points reach or exceed the activation threshold:
    - The skill becomes fully active at its maximum value
    - Any additional points beyond the threshold are effectively ignored
    - This behavior integrates seamlessly with our existing maximum skill cap system

This approach ensures group skills are properly managed in our optimization process, only contributing when their activation requirements are met.

![Group skill threshold implementation](./doc/code/group_skill_threshold.png)

We haven't yet created the master skill proxy variables that will unify all skill types in our optimization. This is because we first need to implement the series skill constraints and their corresponding proxy variables.

So far, we have:

- `skill_sum_capped` variables for standard skills
- `group_skill_sum_capped` variables for group skills

However, these proxy variables are not yet used in our objective function, so the group skill constraints we defined are currently inactive.

Once we implement the series skill system, we will create master proxy variables that combine:

- Standard skills
- Group skills
- Series skills

This unified approach will be covered after we complete the series skill constraint implementation.

### Series skills

Series skills are like group skills with a difference:
while group skills have only one activation threshold, series skills have multiple ones. The constraint management here is more complex.

For example, a series skill might have thresholds at:

- 2 points - Basic activation
- 4 points - Enhanced effect

At each threshold, the skill provides different benefits:

- 0-1 points: No effect
- 2-3 points: Level 1 activation
- 4-5 points: Level 2 activation  

This creates a stepped activation pattern where the skill's effectiveness increases at specific point thresholds rather than the single threshold of group skills. Our optimization needs to handle these multiple activation levels while still respecting the overall skill point caps.

The implementation of series skills involves several key components:

1. For each series skill
2. We iterate over threshold ranges (e.g. 0->2, 2->4)
3. When the skill level is within this range, we force it to be the value of the lower threshold

To check if the value is in range, we must go through multiple proxy variables:

1. We create a variable that indicates if it is higher than lower bound
2. We create a variable that indicates if it is lower than upper bound
3. We create a variable that indicates both preceding variables are true. This way we have a "is between" boolean variable
4. If this "is between" boolean variable is true, we set the skill value to the lower threshold value

One key aspect here is that we have multiple pairwise thresholds. So, what we do here is:

1. For each threshold pair, we have a variable which is equal to the lower threshold if the skill sum is within range
2. This same variable == 0 if the skill sum is not within range
3. We consolidate all these individual threshold-skill-sums into a single sum with the `series_skill_sum_capped` proxy variable.

![Series skill solution implementation](./doc/code/series_skill_solution.png)

### Unifying standard, group, and series skills

We created specialized proxy variables for each skill type (standard, group, and series) to handle their unique mechanics.

To optimize our objective function effectively, we need a unified approach that combines all skill types into a single calculation.

The process of unifying these different skill types is straightforward and involves unifying those variables in the same place:

![Unified skills implementation](./doc/code/unify_skills.png)

### Jewels

Jewels are special items that can be inserted into equipment slots to provide additional skill points.

Each piece of armor has a specific number and type of jewel slots. There are three types of jewel slots (1, 2, and 3) with the following compatibility rules:

- Type 3 jewels can only fit in type 3 slots
- Type 2 jewels can fit in type 2 or 3 slots
- Type 1 jewels can fit in any slot type (1, 2, or 3)

To implement these rules in our optimization model, we need to handle two main components:

1. Slot Management
   - Create variables to track the number of empty slots of each type (1, 2, 3) across all equipped gear
   - Maintain separate counts for each slot type

2. Jewel Assignment
   - Define integer variables for each unique jewel to track how many will be used
   - Add constraints to ensure we don't exceed available slot capacity
   - Include the skills provided by equipped jewels in our skill point calculations

#### Slot management

First, let's prepare the data prior to registering empty jewel slots

![Jewel data preparation](./doc/code/jewel_data_preparation.png)

This gives us a dataset with the following form

| piece   | name                     |   jewel_1 |   jewel_2 |   jewel_3 |
|:--------|:-------------------------|----------:|----------:|----------:|
| Tête    | Heaume Rathalos Gardien  |         0 |         0 |         0 |
| Bras    | Avant-bras Uth Duna      |         1 |         0 |         0 |
| Bras    | Avant-bras Gore Magala β |         0 |         2 |         0 |
| Taille  | Boucle en alliage α      |         1 |         0 |         0 |
| Jambes  | Grèves Blangonga β       |         0 |         2 |         0 |

First, we will prepare the registry entries to store the amount of jewel emplacements we have at our disposal:

![Jewel preparation registry creation](./doc/code/jewel_preparation_create_registry.png)

For each armor piece and jewel type combination:

- We create a variable to track available slots
- We link this to whether the armor piece is equipped
- If equipped: slot count matches the armor's data
- If not equipped: slot count is zero

![Jewel emplacement list registration](./doc/code/jewel_preparation_register_emplacement_lists.png)

Now, we need to aggregate the total number of jewel slots, for each jewel type:

![Jewel emplacement count aggregation](./doc/code/jewel_aggregate_jewel_emplacement_counts.png)

At this point, we have created variables that track the total number of available jewel slots of each type (1, 2, and 3) based on the armor pieces currently equipped.

Now, we need to move on to jewel attribution.

#### Jewel attribution

Jewel System in Monster Hunter Rise:

The jewel system allows players to have access to all existing jewels in the game, and they can use multiple copies of the same jewel if desired.

For jewel slot rules, size 1 jewels are versatile and can be fitted into any slot size (1, 2, or 3). Size 2 jewels are more restricted and can only be placed in size 2 or 3 slots. Size 3 jewels are the most restrictive and can only be fitted into size 3 slots.

The system operates under several constraints. The total number of jewels used cannot exceed the available slot capacity on equipped armor pieces. Each type of jewel has its own usage counter variable to track how many are being used. The system must respect the compatibility rules between jewel sizes and slot sizes when making assignments.

First, we are going to prepare the jewel data:

![Jewel preparation datasheets](./doc/code/jewer_preparation_datasheets.png)

This gives us the following dataset:

| name                  |   jewel_lvl | skill_name           |   skill_lvl |
|:----------------------|------------:|:---------------------|------------:|
| Joyau attaque [1]     |           1 | Machine de guerre    |           1 |
| Joyau attaque II [2]  |           2 | Machine de guerre    |           2 |
| Joyau attaque III [3] |           3 | Machine de guerre    |           3 |
| Joyau vengeance [2]   |           2 | Vengeance            |           1 |
| Joyau riposte [3]     |           3 | Poussée d'adrénaline |           1 |

Now, we will create variables that indicate how many of each jewel we are going to use.

We will also associate the fact that we benefit from skill points from one or multiple jewel usages.

![Jewel registration implementation](./doc/code/jewel_register_each_one.png)

Now, we need to make sure that the amount of jewel uses doesn't exceed the amount of jewel slots.

![Jewel usage constraints](./doc/code/jewel_usage_constraints.png)

## Objective function

Our optimization process follows a hierarchical set of priorities:

1. Primary goal: Maximize the level of all requested skills to their maximum values
2. Secondary goal: When full optimization isn't possible, prioritize maximizing skills with higher weights
3. Tertiary goal: Among valid solutions, select the one that leaves the most jewel slots unused
4. Final goal: If multiple solutions remain, choose the configuration that provides the most additional bonus skills

### Fulfilling asked skills

Now we can declare objective function.

We use an objective function to maximize rather than asking for a strict solution.

This is because we may ask for something that is not possible to achieve fully.

In the case that we are not able to achieve the max skill level on all asked skills, we may want to maximize levels of one skill before another one.

We will add weights in our objective function in order to achieve that:

![](./doc/code/objective_only_skill.png)

### Maximizing the number of free jewel emplacements

Here, we are going to add a bonus to the objective function in order to find solutions that maximize the number of free jewel-3 emplacements, then the number of free jewel-2 emplacements, and finally the number of free jewel-1 emplacements.

![](./doc/code/objective_free_jewel_emplacements.png)

### Maximizing the number of bonus skills

Finally, if there are still multiple solutions, prioritize the ones that provide more additional skills.

![](./doc/code/total_skill_points.png)

### Building the objective function

To ensure each sub-objective is strictly prioritized over the next one, we multiply each objective by different powers of 10. This creates a clear hierarchy where even the maximum value of a lower priority objective cannot influence the outcome of a higher priority objective.

![](./doc/code/final_objective_function.png)

# Conclusion

In this exploration of optimization techniques for Monster Hunter equipment loadouts, we've developed an approach to solving a complex multi-objective optimization problem. Our solution leverages constraint programming to handle the intricate constraints of armor and decoration combinations while maintaining a clear hierarchy of optimization goals.

The key innovations in our approach include:

1. A flexible constraint system that accurately models the game's equipment rules, including armor piece restrictions and decoration slot limitations

2. A hierarchical objective function that prioritizes different aspects of the build:
   - Primary focus on achieving desired skill levels
   - Secondary consideration for skill priority weights
   - Optimization of jewel slot efficiency
   - Maximization of bonus skills as a final criterion

This approach not only offers practical benefits for Monster Hunter players but also illustrates how gaming optimization problems can be tackled using mathematical programming techniques. The method we've crafted could be adapted to similar challenges in other games or fields where resource allocation and multiple objectives need to be balanced.

The implementation demonstrates that even complex gaming optimization problems can be addressed effectively when accurately modeled with the appropriate mathematical tools and optimization strategies.

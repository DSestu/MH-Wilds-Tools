# Recherche d'équipements optimisés

Cet outil permet d'optimiser des combinaisons d'équipements et de joyeaux en fonction de critères de talents désirés.

Pour se faire, une recherche opérationnelle est effectuée, aussi appelée programmation par contraintes.

La totalité de la logique d'équipement de MH Wilds a été implémentée sous forme de contraintes numériques.

L'équipement proposé par l'algorithme est celui qui maximise une fonction de score.

## Fonction de score

La fonction de score est définie de la manière suivante, par ordre de priorité :

1. Maximiser le nombre de points de talents désirés présents dans l'équipement, talisman et joyaux.

2. Maximiser le nombre d'emplacements de joyeaux vides. En priorité de taille 3, puis 2, puis 1.

3. Maximiser le nombre total de points de talents. Cela a pour impact d'avoir d'éventuels talents supplémentaires.

> **Important:** La recherche d'équipement n'est pas stricte: si trop de talents sont désirés, une solution optimale, mais n'ayant pas forcément le maximum de points de talents pour tous les talents, seraa proposée.

Pour cette raison, un système de préférence/poids sur chaque talent est disponible, permettant de favoriser la maximmisation de certains talents en priorité.

## Mode d'emploi

* Ajouter autant de talents que voulus grace au Dropdown si dessous. Une nouvelle ligne sera ajoutée par talent. **Note:** Taper des caractères dans la barre de recherche permet de filtrer les talents affichés.

* Eventuellement mettre une priorité. Plus le nombre est grand, plus le talent sera prioritaire.

* Eventuellement mettre une pénalisation sur le niveau max. Cela est utile pour faire coexister plusieurs bonus d'ensemble d'équipement. Exemple: Gore magala à max 2, et Anjanath à max 2 (au lieu de 4 chacun). C'est aussi utile pour des talents où nous voulons avoir uniquement un seul niveau (exemple: Adaptabilité).

* Selectionner l'arme désirée par le dropdown de classe, puis d'arme.

* Appuyer sur le bouton "Optimiser"

* Profit.

EVO Projekt - Genetické programování - řízení mravence
xtomec09

Založeno na tiny_gp_plus.py https://github.com/moshesipper/tiny_gp/tree/master

Spuštění:
    python ./proj cesta_k_souboru_s_mapou

pokud není cesta_k_souboru_s_mapou je využita mapa 8x8 s cestou

Pro vygenerování map je přiložen skript gen_mazes.py (spuštění: python ./gen_mazes.py)
Který vygeneruje 5 map - 4 použité v experimentech a také 32x32_Santa_Fe_ant_trail (viz https://arxiv.org/pdf/1312.1858 -- článek ze zadání)

Využívá tyto knihovny:
    - random
    - statistics
    - copy
    - matplotlib
    - IPython
    - graphviz
    - pickle
    - pandas
    - os
    - argparse
    - pathlib


Konstanty:
    - POP_SIZE: The population size (default is 200).
    - MIN_DEPTH: The minimal initial random tree depth (default is 2).
    - MAX_DEPTH: The maximal initial random tree depth (default is 5).
    - GENERATIONS: The maximal number of generations to run evolution (default is 1000).
    - TOURNAMENT_SIZE: The size of tournament for tournament selection (default is 3).
    - XO_RATE: The crossover rate (default is 0.8).
    - PROB_MUTATION: The per-node mutation probability (default is 0.2).
    - BLOAT_CONTROL: Adds bloat control to fitness function if set to True (default is True).
    - BLOAT_PENALTY: The penalty for each extra node in a tree (default is 0.0001).
    - ANT_START_ENERGY: The starting energy for the ant (default is 20).
    - ANT_FOOD_ENERGY_GAIN: The energy gain for the ant when it finds food (default is 5).
    - BEST_PARENTS_TO_SURVIVE: The best percentage of the population that survives (default is 5% of POP_SIZE).
    - SHOW_WINDOW: Show window with results (default is True).
    - OUTPUT_DIR: Output directory for results (default is './output').
    - RUN_EXPERIMENT: Number of repetitons of experiment (default is 1).


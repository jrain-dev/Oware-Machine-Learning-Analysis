# Oware AI Competition & Machine Learning Analysis

A self-contained research playground for the traditional West African mancala game **Oware** (also known as Awale, Ayo, or Wari). The project implements the game engine from scratch in NumPy, a spectrum of AI agents ranging from random play to Deep Q-Networks, and the simulation/tournament/analysis tooling needed to compare them statistically.

The repository doubles as the codebase behind an accompanying research write-up (`Oware Research Paper.pdf`), so the engine, logging format, and agent set are built to produce reproducible experimental data rather than just a playable game.

## Table of Contents

- [About Oware](#about-oware)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Game Engine](#game-engine)
- [Game Variants](#game-variants)
- [AI Agents](#ai-agents)
- [Training Deep Q-Network Agents](#training-deep-q-network-agents)
- [Simulation, Tournaments & Analysis](#simulation-tournaments--analysis)
- [Data Output Reference](#data-output-reference)
- [Testing](#testing)
- [Research Paper](#research-paper)
- [Extending the Project](#extending-the-project)
- [Contributing](#contributing)
- [License](#license)

## About Oware

Oware belongs to the mancala family of pit-and-pebble games. Two players each control six pits; on a turn, a player empties one of their pits and "sows" its seeds counter-clockwise around the board, one seed per pit, skipping the originating pit. If the last seed sown lands in an opponent pit that then holds two or three seeds, the mover captures it — and can keep capturing backward through consecutive opponent pits that also hold two or three seeds ("chain captures," depending on the variant). The game ends when a player passes the 24-seed majority threshold or when a side has no legal moves, at which point any remaining seeds are awarded to the opponent still able to move.

This makes Oware an appealing but non-trivial benchmark for game-playing AI: the branching factor is small enough for exhaustive search techniques like minimax to be tractable at shallow depth, while the delayed, chain-reaction nature of captures gives learning-based agents (Q-learning, DQN) something genuinely non-obvious to discover.

## Repository Layout

```
Oware-Machine-Learning-Analysis/
├── owareEngine.py              # OwareBoard: core rules, sowing, captures, win detection
├── agents.py                   # All AI agent implementations (rule-based, RL, deep RL)
├── requirements.txt            # Python dependencies
├── TRAINING_GUIDE.md           # In-depth documentation for the DQN training pipeline
├── test_advanced_analysis.py   # Validation tests for the analysis/statistics tooling
├── Oware Research Paper.pdf    # Written report describing the methodology and findings
├── actions/                    # Application layer: everything you actually run
│   ├── menu.py                 #   Interactive CLI entry point
│   ├── simulation.py           #   Head-to-head games, batch simulation, tournaments
│   ├── analysis.py             #   Statistical analysis over simulation/tournament logs
│   └── training.py             #   Configurable DQN training loop, checkpoints, curricula
└── output/                     # Generated at runtime — CSV logs, checkpoints, reports
    ├── sim_log.csv
    ├── tourney_log.csv
    ├── analysis_log.csv
    └── training/
        └── <session_name>/
            ├── config.json
            ├── training_log.json
            ├── final_metrics.pkl
            └── checkpoints/
```

`owareEngine.py` and `agents.py` are intentionally decoupled from the CLI: any script can `import owareEngine` and `import agents` to build custom experiments without touching `actions/`.

## Installation

### Prerequisites

- Python 3.7+
- NumPy (required — the board state and Q-tables are NumPy arrays)
- Pandas, SciPy, Matplotlib, Seaborn (required for the analysis tooling)
- PyTorch (optional — enables the GPU-capable DQN implementation; a pure-NumPy DQN fallback is used automatically if PyTorch is not installed)

### Setup

```bash
git clone https://github.com/jrain-dev/Oware-Machine-Learning-Analysis.git
cd Oware-Machine-Learning-Analysis
pip install -r requirements.txt
```

`requirements.txt` pins:

```
numpy>=1.19.0
pandas>=1.2.0
torch>=1.9.0        # optional, DQN agents fall back to NumPy without it
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Quick Start

The interactive menu is the fastest way to explore the project:

```bash
cd actions
python menu.py
```

It offers:

1. Run simulations
2. Conduct tournaments
3. Analyze results
4. Train DQN models
5. Exit

Alternatively, drive things directly from Python:

```python
from owareEngine import OwareBoard
from agents import RandomAgent, HeuristicAgent

board = OwareBoard(variant="standard")
agents = [RandomAgent(), HeuristicAgent()]

while not board.game_over:
    player = board.current_player
    valid_moves = board.get_valid_moves(player)
    move = agents[player].select_action(board, valid_moves)
    reward, state, done = board.apply_move(move)

print(board)
print("Winner:", board.winner)  # 0, 1, or -1 for a tie
```

## Game Engine

`owareEngine.py` defines a single class, `OwareBoard`, that holds the entire game state and rule set:

- **Board representation** — a flat NumPy array of 12 pits (`0–5` belong to player 0, `6–11` to player 1), plus a two-element `scores` list.
- **`get_valid_moves(player)`** — returns the indices of that player's non-empty pits.
- **`apply_move(move)`** — empties the chosen pit, sows seeds counter-clockwise (skipping the source pit), triggers `_handle_capture`, flips `current_player`, checks for game end, and returns `(reward, state, done)` where `reward` is `+1`/`-1`/`0` from player 0's perspective — convenient for feeding straight into an RL training loop.
- **`_handle_capture(last_pit)`** — captures the landing pit if it ends on the opponent's side with exactly 2 or 3 seeds, then (variant permitting) walks backward capturing consecutive opponent pits that also hold 2 or 3 seeds.
- **`_check_game_over()`** — ends the game once a player's score exceeds 24 (majority of 48 seeds), or once the side to move has no legal pit to play, in which case all seeds remaining on the board are awarded to the *other* player before a winner is declared. Ties are represented by `winner = -1`.
- **`__str__`** — renders a human-readable two-row board with both players' scores, handy for debugging in a REPL.

## Game Variants

`OwareBoard(variant=...)` supports four rule sets, each of which only changes the starting seed count and whether chain captures are enabled — the capture, sowing, and win-condition logic is shared:

| Variant | Seeds per pit | Total seeds | Chain captures |
| --- | --- | --- | --- |
| `standard` (default) | 4 | 48 | Yes |
| `sparse` | 2 | 24 | Yes |
| `dense` | 6 | 72 | Yes |
| `no_chain` | 4 | 48 | No — only the single landing pit can be captured |

```python
board = OwareBoard(variant="dense")
```

## AI Agents

All agents in `agents.py` implement a common interface — `select_action(board, valid_moves)` — so they can be swapped into simulations, tournaments, or training loops interchangeably. Learning agents additionally expose `update()`/`train_step()`, `end_episode()`, and `save_checkpoint()`/`load_checkpoint()`.

### Rule-based agents

| Agent | Strategy |
| --- | --- |
| **`RandomAgent`** | Picks uniformly at random among legal moves. Baseline for win-rate comparisons. |
| **`GreedyAgent`** | Simulates each candidate move on a deep copy of the board and plays whichever yields the largest immediate increase in total captured seeds. |
| **`HeuristicAgent`** | Strongly prefers any move that captures seeds this turn; if no capture is available, it looks one ply further and picks the move that *minimizes* the opponent's best immediate capture on their following turn. |

### Search-based agent

| Agent | Strategy |
| --- | --- |
| **`MinimaxAgent(depth=2)`** | Depth-limited minimax over `(scores[0] - scores[1])` as the evaluation function, recursively simulating moves on board copies. *Note: the current implementation is plain minimax — it explores the full move tree at the configured depth without alpha-beta pruning cutoffs.* Depth is configurable at construction time. |

### Reinforcement-learning agent

| Agent | Strategy |
| --- | --- |
| **`QLearningAgent`** | Tabular Q-learning keyed on the raw board state tuple (`defaultdict` of 12-length NumPy arrays). Supports configurable learning rate, discount factor, and ε-greedy exploration with decay, plus `save_checkpoint`/`load_checkpoint` via pickle. |

### Deep reinforcement-learning agents

`DQNAgent` (and its three preset sizes `DQNSmall`, `DQNMedium`, `DQNLarge`) implement a standard DQN: a feed-forward Q-network, a periodically-synced target network, an experience replay buffer, and ε-greedy exploration with decay. Invalid actions are masked out with a large negative value before taking the arg-max.

| Preset | Hidden layers | Learning rate | Batch size | Buffer size | Target sync |
| --- | --- | --- | --- | --- | --- |
| **DQNSmall** | (64,) | 1e-3 | 32 | 5,000 | every 20 steps |
| **DQNMedium** | (128, 64) | 5e-4 | 64 | 10,000 | every 10 steps |
| **DQNLarge** | (256, 128) | 3e-4 | 128 | 20,000 | every 5 steps |

Two backends share the exact same interface:

- **PyTorch backend** (`DQNNet` + `torch.optim.Adam`) — used automatically whenever `import torch` succeeds. Runs on CUDA if available.
- **Pure NumPy backend** (`NumpyDQNAgent`) — a hand-written one- or two-hidden-layer MLP with manual forward pass and backpropagation (ReLU activations, MSE loss, SGD-style updates), automatically substituted in whenever PyTorch is not installed. This means every DQN experiment in the repo can still be run with zero deep-learning dependencies, just more slowly.

Both DQN implementations track running training statistics (`training_losses`, `episode_rewards`, `epsilon`) and can serialize/deserialize full training state (`save_checkpoint`/`load_checkpoint` — `.pth` for PyTorch, `.npz` for NumPy).

```python
from agents import QLearningAgent, MinimaxAgent, DQNMedium

q_agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, exploration_rate=0.8)
minimax_agent = MinimaxAgent(depth=3)
dqn_agent = DQNMedium()  # falls back to NumPy automatically if torch isn't installed
```

## Training Deep Q-Network Agents

The `actions/training.py` module implements a full training pipeline on top of `DQNAgent`/`NumpyDQNAgent`: configurable episode counts, periodic evaluation against a curriculum of opponents, checkpointing (including automatic "best model" tracking), and early stopping. Full documentation lives in [`TRAINING_GUIDE.md`](./TRAINING_GUIDE.md); the highlights:

**Menu-driven:**

```bash
cd actions
python menu.py
# Option 4 (train dqn) → Option 1 (Quick Training) or Option 2 (Advanced Training)
```

**Command line:**

```bash
cd actions
python3 training.py small 2000                       # train DQNSmall for 2000 episodes
python3 training.py medium 5000                       # train DQNMedium for 5000 episodes
python3 training.py large 10000                       # train DQNLarge for 10000 episodes
python3 training.py evaluate checkpoint.pth small 100 # evaluate a saved checkpoint
python3 training.py list                              # list past training sessions
```

**Programmatic configuration:**

```python
from actions.training import create_training_config, DQNTrainer
from agents import DQNMedium

config = create_training_config(
    total_episodes=10000,
    eval_interval=200,
    checkpoint_interval=400,
    patience=1500,
    variant='dense',
    opponent_weights=[0.3, 0.3, 0.3, 0.1],  # tougher opponent mix
)

trainer = DQNTrainer(config)
results = trainer.train(DQNMedium, 'my_advanced_training')
```

**Default training curriculum** — the trainer mixes opponents so the agent doesn't overfit to one strategy:

| Opponent | Default weight |
| --- | --- |
| Random | 40% |
| Greedy | 30% |
| Heuristic | 20% |
| Minimax | 10% |

**Key `TrainingConfig` fields:**

| Parameter | Default | Meaning |
| --- | --- | --- |
| `total_episodes` | 5000 | Total training episodes |
| `warmup_episodes` | 100 | Episodes before training/learning starts |
| `eval_interval` | 250 | Episodes between evaluation passes |
| `eval_episodes` | 50 | Episodes per evaluation |
| `checkpoint_interval` | 500 | Episodes between checkpoints |
| `patience` | 1000 | Early-stopping patience |
| `variant` | `standard` | Which `OwareBoard` variant to train on |

Every session writes to `output/training/<session_name>/`, including `config.json`, an episode-by-episode `training_log.json`, a `final_metrics.pkl`, and a `checkpoints/` directory with versioned `.pth`/`.npz` files, per-checkpoint metadata, and a symlinked `best_checkpoint.pth`.

## Simulation, Tournaments & Analysis

`actions/simulation.py` and `actions/analysis.py` provide the batch-experiment layer on top of the engine and agents.

**Batch simulation** (agent vs. agent, N episodes, results logged to CSV):

```bash
cd actions
python simulation.py
```

**Tournaments** (single-elimination bracket across every registered agent):

```bash
python -c "import simulation; simulation.run_tournament()"
```

Both are also reachable from `menu.py` (options `1` and `2`).

**Statistical analysis** over the resulting logs (win rates by agent, average game length, capture statistics, head-to-head breakdowns):

```bash
python menu.py   # option 3: run analysis
```

Run simulations/tournaments first — analysis reads from `output/sim_log.csv` and `output/tourney_log.csv` and writes its findings to `output/analysis_log.csv`.

## Data Output Reference

Everything the project produces lands in `output/` as CSV (simulation/tournament/analysis) or structured training artifacts (JSON/pickle/checkpoint files):

- **`sim_log.csv`** — episode number, agent match-up, winner, final scores, move count, game length, capture counts, and agent-specific fields (e.g. the acting ε for RL agents).
- **`tourney_log.csv`** — match IDs, participants, best-of-series results, aggregate match statistics, bracket progression.
- **`analysis_log.csv`** — per-agent win rates, summary statistics, and comparison metrics computed from the two logs above.
- **`training/<session>/`** — see [Training Deep Q-Network Agents](#training-deep-q-network-agents).

## Testing

`test_advanced_analysis.py` at the repo root exercises the statistical analysis pipeline to guard against regressions in how win rates, aggregate scores, and trend metrics are computed. Run it with your test runner of choice, e.g.:

```bash
python -m pytest test_advanced_analysis.py
# or
python test_advanced_analysis.py
```

## Research Paper

`Oware Research Paper.pdf` is the written report that accompanies this codebase — it documents the experimental methodology, agent comparisons, and findings produced using this engine and training pipeline. Refer to it for the motivation and analysis behind the agent designs above.

## Extending the Project

Ideas for further work, several of which are also called out in `TRAINING_GUIDE.md`'s "Future Enhancements" section:

1. **New AI strategies** — Monte Carlo Tree Search, genetic/evolutionary algorithms, alpha-beta pruning or iterative deepening for `MinimaxAgent`.
2. **Deeper statistical analysis** — significance testing, Elo-style rating systems across tournament results.
3. **Resume-from-checkpoint training** and **distributed/multi-process training**.
4. **Training visualization** — plots of win rate, loss, and ε over time.
5. **A GUI or web front end** for human-vs-agent play.
6. **Networked multiplayer.**

## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure the existing test suite passes (`test_advanced_analysis.py`)
5. Submit a pull request

## License

This project is open source. See the repository for full license details.
- Traditional Oware game rules and variants
- PyTorch community for deep learning frameworks
- Reinforcement learning research community

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/jrain-dev/Oware-AI-Competition-Bot-Game).

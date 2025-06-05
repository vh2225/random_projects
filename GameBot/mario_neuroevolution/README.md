# Mario Neuro-Evolution

This project uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural networks that can play Super Mario Bros.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new neural network:

```bash
python main.py --mode train --generations 100 --level 1-1
```

Options:
- `--generations`: Number of generations to evolve (default: 100)
- `--level`: Mario level to train on (default: 1-1)
- `--render`: Show game window during training (slower but visual)
- `--checkpoint`: Resume training from a checkpoint file
- `--checkpoint-freq`: How often to save checkpoints (default: 10)

### Playing with a Trained Model

To play using a trained genome:

```bash
python main.py --mode play --genome outputs/run_*/winner_genome.pkl --level 1-1
```

## Project Structure

- `main.py`: Main training and playing script
- `configs/neat_config.txt`: NEAT algorithm configuration
- `agents/mario_agent.py`: Neural network agent for Mario
- `utils/fitness.py`: Fitness evaluation functions
- `utils/visualization.py`: Visualization and plotting utilities
- `outputs/`: Generated visualizations and saved genomes
- `checkpoints/`: Training checkpoints

## How It Works

1. **State Processing**: The game screen is converted to grayscale, resized, and flattened into a 240-dimensional input vector.

2. **Neural Network**: NEAT evolves both the topology and weights of neural networks. Networks start simple and complexify over generations.

3. **Actions**: The network outputs control 12 possible actions (the SIMPLE_MOVEMENT action space).

4. **Fitness Function**: Agents are evaluated based on:
   - Distance traveled (primary metric)
   - Coins collected
   - Score achieved
   - Efficiency (distance per step)
   - Level completion bonus

5. **Evolution**: The NEAT algorithm:
   - Selects the fittest individuals
   - Applies mutations (add nodes, add connections, change weights)
   - Maintains species to protect innovation
   - Crosses over successful genomes

## Configuration

Key parameters in `configs/neat_config.txt`:
- `pop_size`: Population size (150)
- `num_inputs`: Input neurons (240 - processed game state)
- `num_outputs`: Output neurons (12 - possible actions)
- `fitness_threshold`: Target fitness to stop evolution
- Various mutation rates and probabilities

## Tips for Better Performance

1. **Longer Training**: More generations typically lead to better performance
2. **Population Size**: Larger populations explore more solutions but train slower
3. **Action Space**: Simpler action spaces (RIGHT_ONLY) train faster but are less capable
4. **Multiple Levels**: Train on different levels for more robust agents
5. **Fitness Tuning**: Adjust fitness weights based on desired behavior

## Troubleshooting

- If training is too slow, reduce `pop_size` or disable rendering
- If agents get stuck, adjust the stuck detection threshold in fitness.py
- For better exploration, increase mutation rates in the config
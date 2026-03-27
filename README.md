# ALPHACODE

**ALPHACODE** - MCTS-based Code Exploration with Git

```
      .o.       oooo             oooo                    .oooooo.                   .o8            
     .888.      `888             `888                   d8P'  `Y8b                 "888            
    .8"888.      888  oo.ooooo.   888 .oo.    .oooo.   888           .ooooo.   .oooo888   .ooooo.  
   .8' `888.     888   888' `88b  888P"Y88b  `P  )88b  888          d88' `88b d88' `888  d88' `88b 
  .88ooo8888.    888   888   888  888   888   .oP"888  888          888   888 888   888  888ooo888 
 .8'     `888.   888   888   888  888   888  d8(  888  `88b    ooo  888   888 888   888  888    .o 
o88o     o8888o o888o  888bod8P' o888o o888o `Y888""8o  `Y8bood8P'  `Y8bod8P' `Y8bod88P" `Y8bod8P' 
                       888                                                                         
                      o888o                                                                        
```

## Features

- **MCTS Search** - Monte Carlo Tree Search for code exploration
- **Git Branches** - Each node is a git commit for easy rollback
- **Island Model** - Multiple independent search islands with migration
- **Feature Grid** - MAP-Elites style diversity preservation
- **Intent Detection** - Smart filtering of non-code requests
- **Entropy Analysis** - Adaptive branch count based on model confidence
- **Light Evaluation** - Fast local checks + quick LLM scoring

## Installation

```bash
# Clone repository
git clone https://github.com/example/alphacode.git
cd alphacode

# Install with uv
uv sync
```

## Usage

```bash
# Start interactive CLI
uv run alphacode

# Or specify working directory
uv run alphacode -d /path/to/project
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/status` | Show search status |
| `/tree` | Show search tree |
| `/best` | Show best solution |
| `/alternatives` | Show alternative solutions |
| `/adopt` | Merge best solution to main |
| `/quit` | Exit |

## Configuration

Edit `config.yaml`:

```yaml
# Search
max_iterations: 10
max_depth: 10
num_islands: 2

# LLM
llm:
  model: meta/llama-3.1-8b-instruct
  api_base: https://integrate.api.nvidia.com/v1
  api_key: your-key

# Branch strategy
num_actions_per_expand: 2
intent_check: true
```

## Architecture

```
alphacode/
├── cli.py           # Interactive CLI
├── config.py        # Configuration
├── core/
│   ├── controller.py    # Main MCTS controller
│   ├── node.py          # Search nodes
│   └── tree.py          # Search tree + Feature Grid
├── llm/
│   ├── client.py        # LLM API client
│   ├── prompts.py       # Prompt templates
│   ├── intent.py        # Intent detection
│   └── entropy.py       # Entropy analysis
├── state/
│   └ git_manager.py     # Git state management
├── tools/
│   └ executor.py        # Tool execution
├── evaluation/
│   └ evaluator.py       # Code evaluation
└── utils/
    └ display.py         # CLI display utilities
```

## LLM Efficiency

| Task Type | LLM Calls |
|-----------|-----------|
| Chitchat | 1 |
| Simple task | ~5 |
| Complex task | ~8 |

Much more efficient than naive approaches (20-30 calls).

## License

MIT
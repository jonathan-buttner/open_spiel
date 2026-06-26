SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PYTHONPATH="$SCRIPT_DIR/build/python:$PYTHONPATH"
export FORCETEKI_PATH=/Users/jbuttner/proj/home/forceteki
export PYTHONHASHSEED=1

# ./venv/bin/python3 open_spiel/python/examples/forceteki_psro.py \
#   --game_name=python_forceteki_swu \
#   --n_players=2 \
#   --seed=1 \
#   --oracle_type=PPO \
#   --gpsro_iterations=5 \
#   --number_training_episodes=100 \
#   --sims_per_entry=20 \
#   --verbose=True \
#   --debug=True \
#   --parallel_training_workers=4 \
#   --parallel_eval_workers=4 \
#   --output_dir=forceteki_runs/run_001 \
#   --deck_pool_path=/Users/jbuttner/proj/home/swu-meta/data/decks

./venv/bin/python3 open_spiel/python/examples/forceteki_psro.py \
  --game_name=python_forceteki_swu \
  --n_players=2 \
  --seed=1 \
  --oracle_type=PPO \
  --gpsro_iterations=5 \
  --number_training_episodes=100 \
  --sims_per_entry=20 \
  --verbose=True \
  --parallel_training_workers=4 \
  --parallel_eval_workers=4 \
  --output_dir=forceteki_runs/run_001 \
  --meta_strategy_method=nash \
  --debug=minimal \
  --deck_pool_path=/Users/jbuttner/proj/home/swu-meta/data/decks


# --resume_from=forceteki_runs/run_001 \
# ./venv/bin/python3 open_spiel/python/examples/forceteki_psro.py \
#   --game_name=python_forceteki_swu \
#   --n_players=2 \
#   --seed=1 \
#   --oracle_type=PPO \
#   --gpsro_iterations=2 \
#   --verbose=True \
#   --parallel_training_workers=4 \
#   --parallel_eval_workers=4 \
#   --output_dir=forceteki_runs/run_001 \
#   --deck_pool_path=/Users/jbuttner/proj/home/swu-meta/data/decks

export PYTHONPATH=/Users/jbuttner/proj/home/open_spiel:$PYTHONPATH
export PYTHONPATH=/Users/jbuttner/proj/home/open_spiel/build/python:$PYTHONPATH
export FORCETEKI_PATH=/Users/jbuttner/proj/home/forceteki

./venv/bin/python3 open_spiel/python/examples/forceteki_psro.py \
  --game_name=python_forceteki_swu \
  --n_players=2 \
  --oracle_type=PPO \
  --meta_strategy_method=uniform \
  --gpsro_iterations=2 \
  --number_training_episodes=20 \
  --sims_per_entry=10 \
  --verbose=True \
  --forceteki_worker_pool_size=4 \
  --parallel_eval_workers=4

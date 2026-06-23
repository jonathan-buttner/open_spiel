export PYTHONPATH=/Users/jbuttner/proj/home/open_spiel:$PYTHONPATH
export PYTHONPATH=/Users/jbuttner/proj/home/open_spiel/build/python:$PYTHONPATH
export FORCETEKI_PATH=/Users/jbuttner/proj/home/forceteki
export PYTHONHASHSEED=1

./venv/bin/python3 open_spiel/python/examples/forceteki_psro.py \
  --game_name=python_forceteki_swu \
  --n_players=2 \
  --seed=1 \
  --forceteki_seed=1 \
  --oracle_type=PPO \
  --gpsro_iterations=2 \
  --verbose=True \
  --forceteki_worker_pool_size=4 \
  --parallel_eval_workers=4 \
  --output_dir=forceteki_runs/run_001 \
  --deck_pool_path=/Users/jbuttner/proj/home/swu-meta/data/decks

Grid config generation and submission

- Generate configs:

  python scripts/make_grid_configs.py --base babylm_gpt2_100m.yaml

  This writes configs to `configs/grid/` and a `configs/grid/manifest.tsv` mapping
  numeric index -> config path and out_dir. The manifest is used by `submit_grid.sh`.

- Submit (example):

  sbatch --array=1-48 submit_grid.sh

  Or run a single job locally: `./submit_grid.sh 1`.

Notes:
- The scripts set `train.max_tokens=100_000_000` and `resume=false` in generated configs.
- To reduce intermediate checkpoint clutter the job sets a very large `save_interval` and
  `submit_grid.sh` deletes all `step-*` directories except the last one after training finishes.

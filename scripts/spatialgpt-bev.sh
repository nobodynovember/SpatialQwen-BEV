DATA_ROOT=datasets
outdir=${DATA_ROOT}/exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root datasets/observations
      --split SpatialGPTBEV_72_scenes_processed
      --end 20
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm qwen3-vl-8b
      --response_format json
      --max_tokens 4096
      "

python vln/main_gpt.py $flag

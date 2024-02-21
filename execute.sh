CUDA_VISIBLE_DEVICES="MIG-0b2452d4-9b27-530f-a6f1-1c2d05dfaa72" python main.py \
  --base configs/stable-diffusion/v1-finetune.yaml \
  -t \
  --actual_resume ./models/sd/v1-5-pruned.ckpt\
  -n timestep_embedding_768t_StarEmbedExceptItself_RandomConceptToken_Not4bae_0.3contrastive_FFresidual_0.02temperature_golden_temples \
  --gpus 0, \
  --data_root ./example_images/golden_temples_folder
```bash
export CHALLENGE_CONFIG_FILE=configs/challenge_objectnav2022.local.rgbd.yaml
python Stubborn/eval.py --timestep_limit 451 --evaluation local -v 2 --print_images 1 --use_semantics -d tmp2 [--use_lm]
```

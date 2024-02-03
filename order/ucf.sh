# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split1-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split1-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split1-itm.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split2-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split2-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split2-itm.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split3-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split3-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ucf-split3-itm.py 2 --validate --seed 42 --deterministic

CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ntu-xview-raw.py 2 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ntu-xview-itm.py 2 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/plus/ntu-xview-pacl.py 2 --validate --seed 42 --deterministic
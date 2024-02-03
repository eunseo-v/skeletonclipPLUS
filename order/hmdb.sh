# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split1-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split1-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split1-itm.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split2-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split2-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split2-itm.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split3-raw.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split3-pacl.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split3-itm.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split1-poseconv3d.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split2-poseconv3d.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/hmdb-split3-poseconv3d.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ucf-split1-poseconv3d.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ucf-split2-poseconv3d.py 2 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ucf-split3-poseconv3d.py 2 --validate --seed 42 --deterministic

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ntu-xsub-raw.py 2 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ntu-xsub-itm.py 2 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/plus/ntu-xsub-pacl.py 2 --validate --seed 42 --deterministic
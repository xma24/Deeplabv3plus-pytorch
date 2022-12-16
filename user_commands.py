""">>> C1: run model的shell命令; """
""">>> 
    - ()bash main_dist_train.sh --config=./configs/config_v0.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v1.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v2.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v3.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v4.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v5.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v6.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v7.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v8.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v9.yaml
    - ()bash main_dist_train.sh --config=./configs/config_v10.yaml
"""


""">>> C2: 开启4个gpus的docker"""
""">>> 
    # GPUs: 0 1 2 3 
    docker run -itd --init --shm-size="300g" \
        --name="uni_cuda_latest_gpu0123" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
        --ipc=host --user="$(id -u):$(id -g)" \
        --volume="/home/xma24:/home/xma24" \
        --volume="/data/SSD1:/data/SSD1" \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        log4maxim/uni_cuda_latest:v python3

    # GPUs: 4 5 6 7
    docker run -itd --init --shm-size="300g" \
        --name="uni_cuda_latest_gpu4567" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
        --ipc=host --user="$(id -u):$(id -g)" \
        --volume="/home/xma24:/home/xma24" \
        --volume="/data/SSD1:/data/SSD1" \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        log4maxim/uni_cuda_latest:v python3

"""

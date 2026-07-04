ENV_NAME=safeil

# sudo apt update
# sudo apt upgrade -y
# sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3  # for mujoco-py

conda create -n $ENV_NAME python=3.7 pip

source activate $ENV_NAME


cd 3rdparty/safety-gym
pip install -e .

cd ../safety-starter-agents
pip install -e .

pip install numpy==1.21.6 protobuf==3.20.3 gast==0.2.2 wandb opencv-python tqdm
conda install -y "ffmpeg<5" x264

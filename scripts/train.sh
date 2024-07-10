cd ~/Documents/repos/Luca_OmniIsaacGymEnvs/omniisaacgymenvs

# for i in $(seq 1 1000)
# do
#     ~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh scripts/rlgames_train.py task=InspireHandRotateCube num_envs=160 checkpoint=/home/yichao/Documents/repos/Luca_OmniIsaacGymEnvs/omniisaacgymenvs/runs/InspireHandRotateCube/nn/InspireHandRotateCube.pth;
# done

# ~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh scripts/rlgames_train.py task=InspireHandRotateCube num_envs=160
~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh scripts/rlgames_train.py task=InspireHandRotateCube num_envs=160 checkpoint=/home/yichao/Documents/repos/Luca_OmniIsaacGymEnvs/omniisaacgymenvs/runs/InspireHandRotateCube/nn/InspireHandRotateCube.pth;
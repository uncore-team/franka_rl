# franka_rl

This code is based on [dm_robotics_panda](https://github.com/JeanElsner/dm_robotics_panda), from [JeanElsner](https://github.com/JeanElsner), and [rl_spin_decoupler](https://github.com/uncore-team/rl_spin_decoupler/tree/main), from [uncore-team](https://github.com/uncore-team).

The intention is to adapt a reinforcement learning environment with HIL (Hardware In the Loop) to [gymnasium](https://gymnasium.farama.org/)'s API in order to being able to use algorithms libraries such as [SB3](https://stable-baselines3.readthedocs.io/en/master/)


## Install

Clone the repo:

    git clone https://github.com/uncore-team/franka_rl.git
    cd franka_rl

Create a python virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install dm_robotics_panda
    pip install gymnasium
    pip install stable-baselines3[extra]

You also need to add rl_spin_decoupler to your workspace (and add it to .gitignore):

    cd franka_rl
    git clone https://github.com/uncore-team/rl_spin_decoupler.git


# Examples

The code based on *rl_spin_decoupler* uses sockets to communicate two scripts. Open two terminals and execute the code:

    cd franka_rl
    source .venv/bin/activate
    cd test/side_to_side

On terminal 1:

    python baselines_side.py


On terminal 2:

    python panda_side.py --gui

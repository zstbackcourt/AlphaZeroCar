# AlphaZeroCar
This is AlphaZero using MCTS for the AutoDriving to Path Planning

这个版本使用dqn作为policy_value,策略网络的更新使用的是dqn自己的loss，蒙特卡洛树只是用来利用policy采样，即网络的更新不是监督学习是强化学习

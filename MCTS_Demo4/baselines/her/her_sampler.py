import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.
        这里的脚本对应着论文算法中的第三个for循环中的：Sample a set
         of additional goals for replay G:=S(current episode)
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
            采用feature抽样法

        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
            说明：每次随机随机选择 k 个在这个轨迹上并且在这个transition之后的状态作为新目标，
            即如果现在的样本为 (s_t, a_t, s_t+1) ，那么会在 s_t+1, ..., s_T 之间选择 k 个状态对应的目标作为新目标（标记为future）
            这种情况下经验池里面存放的样本数目是真实采样到样本数目的 k+1 倍
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        # 抽样概率为k/(k+1)
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        # 挑选一个episode和一个time step的transitions使用
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        # 挑出选中的transitions，构成一个字典
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        # 选择与概率future_p成比例的future time index。 通过替换未来的目标，这些将用于HER replay。
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) # np.where(condition)输出满足条件的下标
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        # 用future time index选出的goal替换原来的goal，对于其他的transitions保留原始goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        # 重新构建info dictionary用于奖励的计算
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        # 由于我们更改了goal所以重新计算奖励
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions

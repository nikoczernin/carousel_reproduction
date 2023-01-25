from environment import ContextualEnvironment
from policies import KLUCBSegmentPolicy, RandomPolicy, ExploreThenCommitSegmentPolicy, EpsilonGreedySegmentPolicy, TSSegmentPolicy, LinearTSPolicy
import argparse
import json
import logging
import numpy as np
import pandas as pd
import time


# List of implemented policies
def set_policies(policies_name, user_segment, user_features, n_playlists):
    # Please see section 3.3 of RecSys paper for a description of policies
    POLICIES_SETTINGS = {
        'random' : RandomPolicy(n_playlists),
        'etc-seg-explore' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 100, cascade_model = True),
        'etc-seg-exploit' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 20, cascade_model = True),
        'epsilon-greedy-explore' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = True),
        'epsilon-greedy-exploit' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.01, cascade_model = True),
        'kl-ucb-seg' : KLUCBSegmentPolicy(user_segment, n_playlists, cascade_model = True),
        'ts-seg-naive' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 1, cascade_model = True),
        'ts-seg-pessimistic' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = True),
        'ts-lin-naive' : LinearTSPolicy(user_features, n_playlists, bias = 0.0, cascade_model = True),
        'ts-lin-pessimistic' : LinearTSPolicy(user_features, n_playlists, bias = -5.0, cascade_model = True),
        # Versions of epsilon-greedy-explore and ts-seg-pessimistic WITHOUT cascade model
        'epsilon-greedy-explore-no-cascade' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = False),
        'ts-seg-pessimistic-no-cascade' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = False)
    }

    return [POLICIES_SETTINGS[name] for name in policies_name]


def run_experiment(
        users_path="data/user_features.csv",
        playlists_path="data/playlist_features.csv",
        output_path="results.json",
        policies="random,ts-seg-naive",
        n_recos=12,
        l_init=3,
        n_users_per_round=20000,
        n_rounds=100,
        print_every=10,
        user_indices=None
    ):

    class Arguments:
        def __init__(self, users_path, playlists_path, output_path, policies,
                     n_recos, l_init, n_users_per_round, n_rounds, print_every,
                     user_indices):
            self.users_path = users_path
            self.playlists_path = playlists_path
            self.output_path = output_path
            self.policies = policies
            self.n_recos = n_recos
            self.l_init = l_init
            self.n_users_per_round = n_users_per_round
            self.n_rounds = n_rounds
            self.print_every = print_every
            self.user_indices = user_indices


    args = Arguments(users_path, playlists_path, output_path, policies, n_recos, l_init, n_users_per_round, n_rounds, print_every, user_indices)

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)

    if args.l_init > args.n_recos:
        raise ValueError('l_init is larger than n_recos')


    # Data Loading and Preprocessing steps

    logger.info("LOADING DATA")
    logger.info("Loading playlist data")
    playlists_df = pd.read_csv(args.playlists_path)

    logger.info("Loading user data\n \n")
    users_df = pd.read_csv(args.users_path)

    def numbers_only(series, replace_with=0, type="float"):
        # replace string values in the df with replace_with
        def num_only(x):
            try:
                float(x)
                return x
            except ValueError:
                return replace_with
        return series.apply(lambda x: num_only(x))
    users_df = users_df.apply(numbers_only, 0)



    # limit the dataset, if user indicces are given
    if args.user_indices is not None:
        users_df = users_df.iloc[args.user_indices]

    n_users = len(users_df)

    n_playlists = len(playlists_df)
    n_recos = args.n_recos
    print_every = args.print_every

    user_features = np.array(users_df.drop(["segment"], axis = 1))
    user_features = np.concatenate([user_features, np.ones((n_users,1))], axis = 1)
    playlist_features = np.array(playlists_df)

    user_segment = np.array(users_df.segment)

    logger.info("SETTING UP SIMULATION ENVIRONMENT")
    logger.info("for %d users, %d playlists, %d recommendations per carousel \n \n" % (n_users, n_playlists, n_recos))

    cont_env = ContextualEnvironment(user_features, playlist_features, user_segment, n_recos)

    logger.info("SETTING UP POLICIES")
    logger.info("Policies to evaluate: %s \n \n" % (args.policies))

    policies_name = args.policies.split(",")
    policies = set_policies(policies_name, user_segment, user_features, n_playlists)
    n_policies = len(policies)

    if args.n_users_per_round is not None:
        n_users_per_round = args.n_users_per_round
    else:
        n_users_per_round = n_users

    n_rounds = args.n_rounds
    overall_rewards = np.zeros((n_policies, n_rounds))
    overall_optimal_reward = np.zeros(n_rounds)


    # Simulations for Top-n_recos carousel-based playlist recommendations

    logger.info("STARTING SIMULATIONS")
    logger.info("for %d rounds, with %d users per round (randomly drawn with replacement)\n \n" % (n_rounds, n_users_per_round))
    start_time = time.time()
    for i in range(n_rounds):
        # Select batch of n_users_per_round users
        user_ids = np.random.choice(range(n_users), n_users_per_round)
        overall_optimal_reward[i] = np.take(cont_env.th_rewards, user_ids).sum()
        # Iterate over all policies
        for j in range(n_policies):
            # Compute n_recos recommendations
            recos = policies[j].recommend_to_users_batch(user_ids, args.n_recos, args.l_init)
            # Compute rewards
            rewards = cont_env.simulate_batch_users_reward(batch_user_ids= user_ids, batch_recos=recos)
            # Update policy based on rewards
            policies[j].update_policy(user_ids, recos, rewards, args.l_init)
            overall_rewards[j,i] = rewards.sum()
        # Print info
        if i == 0 or (i+1) % print_every == 0 or i+1 == n_rounds:
            logger.info("Round: %d/%d. Elapsed time: %f sec." % (i+1, n_rounds, time.time() - start_time))
            logger.info("Cumulative regrets: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(np.sum(overall_optimal_reward - overall_rewards[j]))) for j in range(n_policies)]))


    # Save results

    logger.info("Saving cumulative regrets in %s" % args.output_path)
    cumulative_regrets = {policies_name[j] : list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j in range(n_policies)}
    with open(f"{args.output_path}", 'w') as fp:
        json.dump(cumulative_regrets, fp)





if __name__ == "__main__":
    run_experiment()

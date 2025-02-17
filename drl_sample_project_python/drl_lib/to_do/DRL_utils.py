import tqdm

from contracts import DeepSingleAgentWithDiscreteActionsEnv
import tensorflow as tf
import numpy as np


def episodic_semi_gradient_sarsa(env: DeepSingleAgentWithDiscreteActionsEnv):
    pre_warm = 10
    epsilon = 0.1
    gamma = 0.9
    max_episodes_count = 100
    print_every_n_episodes = 10

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])

    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            chosen_action = None
            chosen_action_q_value = None
            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                chosen_action = None
                chosen_action_q_value = None
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])

                all_q_values = np.squeeze(q.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)
                # q_value = q.predict(np.array([q_inputs]))[0][0]
                # if chosen_action is None or chosen_action_q_value < q_value:
                #     chosen_action = a
                #     chosen_action_q_value = q_value

            if episode_id % print_every_n_episodes == 0:
                print(f'State Description : {s}')
                print(f'Chosen action : {chosen_action}')
                print(f'Chosen action value : {chosen_action_q_value}')

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                q.train_on_batch(np.array([q_inputs]), np.array([target]))
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = q.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
            next_chosen_action_q_value = q.predict(np.array([next_q_inputs]))[0][0]

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            q.train_on_batch(np.array([q_inputs]), np.array([target]))


# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definitions of default hyperparameters."""

import ml_collections


def get_config():
    """Get the default configuration.

    The default hyperparameters originate from PPO paper arXiv:1707.06347
    and openAI baselines 2::
    https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
    """
    config = ml_collections.ConfigDict()
    # Total number of frames seen during training.
    config.total_frames = 100000000 # MANY HERE
    # The learning rate for the Adam optimizer.
    config.learning_rate = 2.5e-4
    # Batch size used in training.
    config.batch_size = 1024 # 1024 HERE
    # Number of agents playing in parallel.
    config.num_agents = 4
    # Number of steps each agent performs in one policy unroll.
    config.actor_steps = 512 # 512 HERE
    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 3 # 3 HERE
    # RL discount parameter.
    config.gamma = 0.99
    # Generalized Advantage Estimation parameter.
    config.lambda_ = 0.95
    # The PPO clipping parameter used to clamp ratios in loss function.
    config.clip_param = 0.1
    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.01
    # Linearly decay learning rate and clipping parameter to zero during
    # the training.
    config.decaying_lr_and_clip_param = True

    config.eval_frequency = 10
    config.checkpoint_frequency = 50

    return config


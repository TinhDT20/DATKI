import os
import time
import argparse
import json
import numpy as np
import torch

from environments.drone_env import QuadRotorEnvBase
from utils.plotting import plot_state_variables
from dataset import raw_states_to_torch
from models.resnet_like_model import Net
from drone_loss import drone_loss_function

ROLL_OUT = 1
ACTION_DIM = 4


class QuadEvaluator():

    def __init__(self, model, mean=0, std=1):
        self.mean = mean
        self.std = std
        self.net = model

    def stabilize(self, nr_iters=1, render=False, max_time=300):
        collect_data = []
        actions = []
        failure_list = list()  # collect the reason for failure
        collect_runs = list()
        with torch.no_grad():
            eval_env = QuadRotorEnvBase()
            for _ in range(nr_iters):
                time_stable = 0
                eval_env.reset()
                # zero_state = np.zeros(20)
                # zero_state[9:13] = 500
                # zero_state[2] = 2
                # eval_env._state.from_np(zero_state)
                current_np_state = eval_env._state.as_np
                stable = True
                while stable and time_stable < max_time:
                    current_torch_state = raw_states_to_torch(
                        current_np_state,
                        normalize=True,
                        mean=self.mean,
                        std=self.std
                    )
                    suggested_action = self.net(current_torch_state)
                    suggested_action = torch.sigmoid(suggested_action)[0]

                    suggested_action = torch.reshape(
                        suggested_action, (-1, ACTION_DIM)
                    )
                    # Print loss
                    # print(
                    #     "loss:",
                    #     drone_loss_function(
                    #         torch.unsqueeze(
                    #             torch.from_numpy(current_np_state).float(), 0
                    #         ), torch.unsqueeze(suggested_action, 0)
                    #     )
                    # )
                    numpy_action_seq = suggested_action.numpy()
                    for nr_action in range(ROLL_OUT):
                        action = numpy_action_seq[nr_action]
                        actions.append(action)
                        # if render:
                        #     # print(np.around(current_np_state[3:6], 2))
                        #     print("action:", np.around(suggested_action, 2))
                        current_np_state, stable = eval_env.step(action)
                        collect_data.append(current_np_state)
                        if not stable:
                            # print("FAILED")
                            # print(current_torch_state)
                            # print("action", action)
                            # print(current_np_state)
                            att_stable, pos_stable = eval_env.get_is_stable(
                                current_np_state
                            )
                            # if att_stable = 1, pos_stable must be 0
                            failure_list.append(att_stable)
                            break
                        # if render:
                        #     print(current_np_state[:3])
                        # count nr of actions that the drone can hover
                        time_stable += 1
                    if render:
                        # print([round(s, 2) for s in current_np_state])
                        eval_env.render()
                        time.sleep(.1)
                collect_runs.append(time_stable)
        if len(failure_list) == 0:
            failure_list = [0]
        act = np.array(actions)
        collect_data = np.array(collect_data)
        print(
            "Position was responsible in ", round(np.mean(failure_list), 2),
            "cases"
        )
        print("avg and std action", np.mean(act, axis=0), np.std(act, axis=0))
        return (
            np.mean(collect_runs), np.std(collect_runs), np.mean(failure_list),
            collect_data
        )


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="test_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join("trained_models", "drone", model_name)

    # load std or other parameters from json
    with open(os.path.join(model_path, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, "model_quad" + args.epoch))
    net.eval()

    evaluator = QuadEvaluator(
        net,
        mean=np.array(param_dict["mean"]),
        std=np.array(param_dict["std"])
    )
    # watch
    _, _, _, collect_data = evaluator.stabilize(nr_iters=1, render=True)
    # compute stats
    success_mean, success_std, _, _ = evaluator.stabilize(
        nr_iters=100, render=False
    )
    print(success_mean, success_std)
    plot_state_variables(
        collect_data, save_path=os.path.join(model_path, "evaluation.png")
    )

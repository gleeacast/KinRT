import numpy as np
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    from pi_model import PI0

    train_config_name, model_name, checkpoint_id, pi0_step, checkpoint_dir = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
        usr_args.get("checkpoint_dir"),
    )
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        checkpoint_dir=checkpoint_dir,
        record_router_info=usr_args.get("record_router_info", False),
        router_info_dir=usr_args.get("router_info_dir"),
        router_info_compress=usr_args.get("router_info_compress", True),
    )


def is_remote_model(model):
    return hasattr(model, "call")


def eval(TASK_ENV, model, observation):
    if is_remote_model(model):
        if model.call(func_name="is_observation_window_empty"):
            model.call(func_name="set_language", obs=TASK_ENV.get_instruction())

        input_rgb_arr, input_state = encode_obs(observation)
        model.call(func_name="update_observation_window_rpc", obs={
            "rgb": input_rgb_arr,
            "state": input_state,
        })

        actions = model.call(func_name="get_action_chunk")

        for action in actions:
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.call(func_name="update_observation_window_rpc", obs={
                "rgb": input_rgb_arr,
                "state": input_state,
            })
        return

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()

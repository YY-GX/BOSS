import numpy as np
from libero.libero.envs import OffScreenRenderEnv

class SequentialEnv(OffScreenRenderEnv):
    """
    For execute several policies in sequence in several envs
    """

    def __init__(self, n_tasks, **kwargs):
        # yy: note - in this class code, task == env

        self.n_tasks = n_tasks

        self.env_ls = []
        self.task_id = None
        self.complete_task = []
        for i in range(n_tasks):
            env_args = {
                "bddl_file_name": kwargs["bddl_file_name"][i],
                "camera_heights": kwargs["camera_heights"][i],
                "camera_widths": kwargs["camera_widths"][i],
            }
            env_instance = OffScreenRenderEnv(**env_args)  # Initialize the superclass with env_args
            self.env_ls.append(env_instance)

    def reset(self):
        self.task_id = 0
        self.env_ls[0].reset()

    def seed(self, seed):
        seed = np.random.seed(seed)
        for env in self.env_ls:
            env.seed(seed)

    def set_init_state(self, init_states):
        return self.env_ls[self.task_id].set_init_state(init_states)

    def get_sim_state(self):
        return self.env_ls[self.task_id].sim.get_state().flatten()

    def step(self, action):
        obs, reward, done, info = self.env_ls[self.task_id].step(action)
        info['is_init'] = False
        if done:
            self.complete_task.append(self.task_id)
            # yy: if current task_id is already the last one, do nothing
            # yy: auto initialize state for each new subtask - Note: still need to do this init for the 1st task manually
            if self.task_id != (self.n_tasks - 1):
                self.task_id += 1
                # self.set_init_state(self.init_states_ls[self.task_id])
                done = False
                info['is_init'] = True
        info['task_index'] = self.task_id
        return obs, reward, done, info

    def close(self):
        for env in self.env_ls:
            env.close()


from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, SequentialEnv
env_args = {
            "bddl_file_name": ["/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.bddl", "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"],
            "camera_heights": [512, 512],
            "camera_widths": [512, 512],
        }

env = SubprocVectorEnv(
    [lambda: SequentialEnv(n_tasks=2, **env_args) for _ in range(3)]
)

env.reset()
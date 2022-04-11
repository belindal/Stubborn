import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
import agent.utils.pose as pu
from constants import coco_categories, hab2coco, hab2name, habitat_labels_r, fourty221, fourty221_ori, habitat_goal_label_to_similar_coco
import copy
from agent.agent_state import Agent_State
from agent.agent_helper import Agent_Helper
import time
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from agent.utils.object_identification import get_prediction
import agent.utils.visualization as vu


class StubbornAgent(habitat.Agent):
    def __init__(self,args,task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.agent_states = Agent_State(args)
        self.agent_helper = Agent_Helper(args,self.agent_states)
        self.last_sim_location = None
        self.device = args.device
        self.first_obs = True
        self.valid_goals = 0
        self.total_episodes = 0
        self.args = args
        self.timestep = 0
        self.low_score_threshold = 0.7
        self.high_score_threshold = 0.9
        # towel tv shower gym clothes
        # use a lower confidence score threshold for those categories
        self.low_score_categories = {13,14,15,19,21}

    def reset(self):
        self.agent_helper.reset()
        self.agent_states.reset()
        self.last_sim_location = None
        self.first_obs = True
        self.step = 0
        self.timestep = 0
        self.total_episodes += 1
        self.abs_goal_locations = None
        self.failure_modes = []

    def compute_tgt_locs(self, observations):
        # compute target locations relative to starting location of agent
        origin = observations['origin']
        rotation_world_start = quaternion_from_coeff(observations['rotation_world_start'])

        goal_positions = [
            quaternion_rotate_vector(rotation_world_start.inverse(), goal_pos - origin)
            for goal_pos in observations['gt_goal_positions']
        ]
        return goal_positions
    
    # def detect_success(self, observations):
    #     """
    #     Detect whether we are in proximity of goal
    #     """
    #     min_goal_pos = 0
        
    #     if self.shortest_path_cache is None:
    #         path = habitat_sim.MultiGoalShortestPath()
    #         if isinstance(position_b[0], (Sequence, np.ndarray)):
    #             path.requested_ends = np.array(position_b, dtype=np.float32)
    #         else:
    #             path.requested_ends = np.array(
    #                 [np.array(position_b, dtype=np.float32)]
    #             )
    #     else:
    #         path = self.shortest_path_cache
    #     path.requested_start = np.array(position_a, dtype=np.float32)
    #     self.pathfinder.find_path(path)
    #     self.shortest_path_cache = path
    #     distance_to_target = path.geodesic_distance

    #     distance_to_target < self._config.SUCCESS_DISTANCE
    #     for goal_pos in observations['gt_goal_positions']:
    #         min_goal_pos = min(min_goal_pos, goal_pos - observations['self_position'])
    #     return min_goal_pos < threshold

    def visualize_step(self, observations, planner_inputs):
        # visualize step
        if self.args.visualize or self.args.print_images:
            planner_inputs['gt_goal_positions'] = observations.get('gt_goal_positions', None)
            self.agent_helper.vis_image = vu.init_vis_image(planner_inputs['goal_name'], None)
            self.agent_helper._visualize(planner_inputs)

    def act(self, observations):
        self.timestep += 1
        # if passed the step limit and we haven't found the goal, stop.
        if (self.timestep > self.args.timestep_limit and self.agent_states.found_goal == False) or self.timestep > 495:
            self.failure_modes.append("too_long")
            # failure mode: did not find object in time
            return {'action': 0, 'success': False, 'failure_modes': self.failure_modes}
        
        #get first preprocess
        # if observations['semantic'].any():
        #     breakpoint()
        goal = observations['objectgoal']
        goal = goal[0]+1
        if goal in self.low_score_categories:
            self.agent_states.score_threshold = self.low_score_threshold
        if observations['semantic'].any():
            breakpoint()
        info = self.get_info(observations)
        t1 = time.time()

        # get second preprocess
        self.agent_helper.set_goal_cat(goal)
        # obs = (already has semantic info)
        obs, info = self.agent_helper.preprocess_inputs(observations['rgb'],observations['depth'],info)
        t2 = time.time()
        # obs : np.concatenate((rgb:3, depth:1, sem_seg_pred:5), axis=2).transpose(2, 0, 1)
        # [(rgb:3,depth:1,sem_seg_pred:5),w,h]
        # sem_seg_pred: [goal_cat,conflict(?),blacklist,whitelist,0]
        # obs : np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        # breakpoint()
        info['goal_cat_id'] = goal
        info['goal_name'] = habitat_labels_r[goal]
        obs = obs[np.newaxis,:,:,:]
        # now ready to be passed to agent states
        obs = torch.from_numpy(obs).float().to(self.device)
        if self.first_obs:
            self.agent_states.init_with_obs(obs,info)
            self.first_obs = False

        visualize_step = False
        if info['goal_name'] in info['objs_in_view']:
            # detects goal in view, record image
            visualize_step = True
        planner_inputs = self.agent_states.upd_agent_state(obs,info)
        t3 = time.time()
        # now get action
        action = self.agent_helper.plan_act_and_preprocess(planner_inputs)
        t4 = time.time()
        # For data collection purpose, collect data to train the object detection module
        if self.args.no_stop == 1 and action['action'] == 0:
            self.agent_states.clear_goal_and_set_gt_map(planner_inputs['goal'])
            return {'action':1}
        # for debugging
        if self.args.do_error_analysis:
            within_goal_proximity = observations['distance_to_goal'] < observations['success_distance']
        if action['action'] == 0:
            item = self.agent_states.goal_record(planner_inputs['goal'])
            # TODO what goes on here??
            stp = get_prediction(item,goal)
            if stp:
                if self.args.do_error_analysis and not within_goal_proximity:
                    # false positive
                    visualize_step = True
                    self.failure_modes.append(f"false_positive: {self.timestep}")
                    action['success'] = False
                    action['failure_modes'] = self.failure_modes
                return action
            else:
                if self.args.do_error_analysis and within_goal_proximity:
                    # false negative
                    visualize_step = True
                    self.visualize_step(observations, planner_inputs)
                    self.failure_modes.append(f"false_negative (img_class right, final_class wrong): {self.timestep}")
                else:
                    # mistake by image classifier, corrected by final classifier (calibration?)
                    # self.visualize_step(observations, planner_inputs)
                    visualize_step = True
                    self.failure_modes.append(f"true_negative (img_class wrong, final_class right): {self.timestep}")
                self.agent_states.clear_goal(
                    planner_inputs['goal'])
                return {'action': 1}
        elif self.args.do_error_analysis and within_goal_proximity:
            # false negative
            visualize_step = True
            self.failure_modes.append(f"false_negative: {self.timestep}")
        t5 = time.time()

        if visualize_step:
            self.visualize_step(observations, planner_inputs)
        # print(f"Times: preprocess: {t2 - t1}, update state: {t3 - t2}, plan/act: {t4 - t3}")
        return action

    def get_info(self, obs):

        info = {}
        dx, dy, do = self.get_pose_change(obs)
        info['sensor_pose'] = [dx, dy, do]

        if self.abs_goal_locations is None:
            self.abs_goal_locations = self.compute_tgt_locs(obs)
            info['goal_locs'] = self.abs_goal_locations
        # set goal
        return info

    def get_sim_location(self,obs):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        nap2 = obs['gps'][0]
        nap0 = -obs['gps'][1]
        x = nap2
        y = nap0
        o = obs['compass']
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self,obs):
        curr_sim_pose = self.get_sim_location(obs)
        if self.last_sim_location is not None:
            dx, dy, do = pu.get_rel_pose_change(
                curr_sim_pose, self.last_sim_location)
            dx,dy,do = dx[0],dy[0],do[0]
        else:
            dx, dy, do = 0,0,0
        self.last_sim_location = curr_sim_pose
        return dx, dy, do






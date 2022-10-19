import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
import agent.utils.pose as pu
from constants import coco_categories, hab2coco, hab2name, habitat_labels, habitat_labels_r, fourty221, fourty221_ori, habitat_goal_label_to_similar_coco, goal_labels
import copy
from agent.agent_state import Agent_State
from agent.agent_helper import Agent_Helper
from agent.utils.object_identification import get_prediction
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.core.benchmark import convert_to_gps_coords, convert_to_world_coords

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
        self.visited_rooms = set()
        self.expgoal_room = None
        self.expgoal_room_bb = None
        # self.room_to_bbs = {}
        self.num_steps_in_each_room = {}
        self.agent_distance_from_goal = []


    def reset(self):
        self.agent_helper.reset()
        self.agent_states.reset()
        self.last_sim_location = None
        self.first_obs = True
        self.step = 0
        self.timestep = 0
        self.total_episodes += 1
        self.visited_rooms = set()
        self.expgoal_room = None
        self.expgoal_room_bb = None
        # self.room_to_bbs = {}
        self.num_steps_in_each_room = {}
        self.agent_distance_from_goal = []
        self.agent_positions = []
    
    def compute_gps_distance(self, pos1, pos2):
        return ((pos1 - pos2)**2).sum()**(1/2)

    def agent_seen_all_objs_in_curr_room(self, observations):
        # proxy: close enough to vicinity of room, or agent is stuck
        room_center = convert_to_gps_coords(self.expgoal_room_bb.mean(-1), observations['start']['position'], observations['start']['rotation'])
        if self.args.goal_switch_mode == "num_steps":
            steps_in_expgoal_room = self.num_steps_in_each_room.get(self.expgoal_room,0)
            return self.num_steps_in_each_room.get(self.expgoal_room,0) > 25
        elif self.args.goal_switch_mode == "proximity":
            distance = self.compute_gps_distance(
                observations['gps'], room_center,
            )
            return distance < 0.75
    
    def get_curr_room(self, observations):
        for room in observations['accessible_rooms']:
            room_bb = observations['room_id_to_info'][room]['bb']
            if (observations['self_abs_position'] >= room_bb[:,0]).all() and (observations['self_abs_position'] <= room_bb[:,1]).all():
                # in room
                return room
        return None


    def agent_stuck(self):
        # return len(self.agent_distance_from_goal) > 50 and (min(self.agent_distance_from_goal) > 3 or compute_position_changes(self.agent_positions[len(self.agent_distance_from_goal):]) < 0.5)
        return self.args.detect_stuck and self.agent_states.stuck

    def act(self, observations):
        self.timestep += 1
        # if passed the step limit and we haven't found the goal, stop.
        if self.timestep > self.args.timestep_limit and not self.agent_states.found_goal:
            return {'action': 0, 'stop_reason': 'timeout'}
        if self.timestep > 495:
            return {'action': 0, 'stop_reason': 'timeout'}
        if self.agent_stuck():
            return {'action': 0, 'stop_reason': 'stuck'}
        if self.args.explore_rooms:
            # if saw all objects in goal room, or agent is stuck; and haven't found goal, reset goal room.
            if self.expgoal_room is not None:
                # cannot access curr_room
                self.agent_distance_from_goal.append(self.compute_gps_distance(observations['gps'], convert_to_gps_coords(
                    self.expgoal_room_bb.mean(-1), observations['start']['position'], observations['start']['rotation'])))
                if (self.agent_seen_all_objs_in_curr_room(observations) or self.agent_stuck()) and not self.agent_states.found_goal:
                    self.visited_rooms.add(self.expgoal_room)
                    self.agent_states.clear_expgoal()
                    self.expgoal_room = None
        #get first preprocess
        goal = habitat_labels[goal_labels[observations['objectgoal'].item()]]
        # goal = goal[0]+1
        if goal in self.low_score_categories:
            self.agent_states.score_threshold = self.low_score_threshold

        info = self.get_info(observations)
        if self.args.explore_rooms:
            curr_room = self.get_curr_room(observations)
            if curr_room is not None and curr_room == self.expgoal_room:
                # num seps in room if room is the goal room
                if curr_room not in self.num_steps_in_each_room:
                    self.num_steps_in_each_room[curr_room] = 0
                self.num_steps_in_each_room[curr_room] += 1

        # get second preprocess
        self.agent_helper.set_goal_cat(goal)
        obs, info = self.agent_helper.preprocess_inputs(observations['rgb'],observations['depth'],info)
        info['goal_cat_id'] = goal
        info['goal_name'] = habitat_labels_r[goal]
        obs = obs[np.newaxis,:,:,:]
        # now ready to be passed to agent states
        obs = torch.from_numpy(obs).float().to(self.device)
        if self.first_obs:
            self.agent_states.init_with_obs(obs,info)
            self.first_obs = False

        planner_inputs = self.agent_states.upd_agent_state(obs,info)
        planner_inputs['env_id'] = observations['env_id']
        # now get action
        action = self.agent_helper.plan_act_and_preprocess(planner_inputs)
        # For data collection purpose, collect data to train the object detection module
        if self.args.no_stop == 1 and action['action'] == 0:
            self.agent_states.clear_goal_and_set_gt_map(planner_inputs['goal'])
            return {'action': 1}
        if action['action'] == 0:
            stp = True
            if self.args.override_mode == "prior":
                # need some process to ensure we haven't explored everything????
                assert self.args.explore_room_order == "lm_prior" or self.args.explore_room_order == "gt_prior"
                room_priors = []
                if curr_room is not None:
                    curr_room_names = observations['room_id_to_info'][curr_room]['name']
                    for room_name in curr_room_names:
                        if room_name == "Unknown": continue
                        room_priors.append(observations['room2score'][room_name])
                stp = len(room_priors) == 0 or max(room_priors) >= 0.5
            if self.args.override_mode == "classifier":
                item = self.agent_states.goal_record(planner_inputs['goal'])
                stp = get_prediction(item,goal)
            if not stp:
                self.agent_states.clear_goal(planner_inputs['goal'])
                return {'action': 1}
            else:
                action['stop_reason'] = 'found goal'
                return action
        return action

    def get_info(self, obs):

        info = {}
        dx, dy, do = self.get_pose_change(obs)
        info['sensor_pose'] = [dx, dy, do]
        info['gt_goal_name'] = obs['gt_goal_name']
        info["gps"] = obs["gps"]

        gt_goal_positions = []
        for g, goal_pos in enumerate(obs['gt_goal_positions']):
            goal_pos_gps = convert_to_gps_coords(goal_pos, obs['start']['position'], obs['start']['rotation'])
            x,y = self.get_sim_location(goal_pos_gps)
            gt_goal_positions.append([x,y])
        info['gt_goal_positions'] = gt_goal_positions

        if self.args.explore_rooms:
            if self.expgoal_room is None:
                possible_goal_rooms = [goal_room for goal_room in obs['goal_rooms'] if goal_room not in self.visited_rooms]
                if len(possible_goal_rooms) == 0:
                    info['expgoal_room_center'] = None
                else:
                    self.expgoal_room = possible_goal_rooms[0]
                    self.expgoal_room_bb = obs['room_id_to_info'][self.expgoal_room]['bb']
            if self.expgoal_room is not None:
                goal_room_gps = convert_to_gps_coords(self.expgoal_room_bb.mean(-1), obs['start']['position'], obs['start']['rotation'])
                x,y = self.get_sim_location(goal_room_gps)
                info['expgoal_room_center'] = (x,y)

        # set goal
        return info

    def get_sim_location(self, gps_coords, compass=None):
        """Returns x, y, o position in internal Stubborn map given gps coordinates."""
        nap2 = gps_coords[0]
        nap0 = -gps_coords[1]
        x = nap2
        y = nap0
        if compass is None:
            return x, y
        else:
            o = compass
            if o > np.pi:
                o -= 2 * np.pi
            return x, y, o

    def get_pose_change(self,obs):
        curr_sim_pose = self.get_sim_location(obs['gps'], obs['compass'])
        if self.last_sim_location is not None:
            dx, dy, do = pu.get_rel_pose_change(
                curr_sim_pose, self.last_sim_location)
            dx,dy,do = dx[0],dy[0],do[0]
        else:
            dx, dy, do = 0,0,0
        self.last_sim_location = curr_sim_pose
        return dx, dy, do






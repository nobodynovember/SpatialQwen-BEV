import os
import logging
import shutil
import sys
import tempfile
import math
import numpy as np
import networkx as nx
import json
import re
import torch


# removed zmq-based BEV IPC; use in-process BEV manager
from bevbuilder.bev_process import start_bev_manager, stop_bev_manager
# import cv2
from GPT.api import gpt_infer

class SpatialExpert:
    def __init__(self, args, env, prompt_manager):
        self.args = args
        self.env = env
        self.prompt_manager = prompt_manager
        self.actionlist = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        self.SpatialKnowledgeGraph = nx.Graph() 
        self.Trajectory = [] 
        self.intersections = [] 
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.dead_end=0
        self.check_down_elevation=False
        self.num_backtrack=0
        self.num_reflect=0
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0
        # start in-process BEV manager (keeps per-scan worker subprocesses)
        try:
            self.bev_manager = start_bev_manager()
        except Exception as e:
            print(f"Warning: failed to start in-process BEV manager: {e}")
            self.bev_manager = None
        # trajectory display bookkeeping: store display-time ids for visited viewpoints
        self.traj_display_id = []
        self.traj_first_visit = {}
        # floor tracking for display bookkeeping: reference Z and threshold (meters)
        self.floor_ref_z = None
        self.floor_delta_thresh = 2.0
        # per-floor display counter: assign display ids starting at 0 after floor change
        self.floor_display_counter = 0
        self.floor_changed = False
        self.semantic_check=True
        self.syn_flag=True
        self.uncertain_flag=False
        self.stop_alternative_list=[]
        self.back_to_stop=False
        self.path_to_alt=None
        self.next_viewpoint_to_stop=None

    def load_gpt_json(self, nav_output, context):
        try:
            return json.loads(nav_output)
        except json.JSONDecodeError as e:
            preview = nav_output
            if len(preview) > 4000:
                preview = preview[:4000] + "...<truncated>"
            print(f"[GPT JSON Parse Error] {context}: {e}", flush=True)
            print(f"[GPT JSON Raw Output] {repr(preview)}", flush=True)
            raise

    def get_strict_json_schema(self, schema_name):
        yes_no = {"type": "string", "enum": ["yes", "no"]}
        short_reason = {"type": "string", "maxLength": 320}
        image_id = {"type": "integer", "minimum": 0, "maximum": 11}
        score_item = {
            "type": "object",
            "properties": {
                "image_id": image_id,
                "similarity_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["image_id", "similarity_score"],
            "additionalProperties": False,
        }
        navigation_alignment = {
            "type": "object",
            "properties": {
                "stop_distance": {"type": "number"},
                "path_plan": {"type": "string", "maxLength": 180},
                "selected_image": image_id,
                "backup_direction_list": {
                    "type": "array",
                    "items": image_id,
                    "minItems": 0,
                    "maxItems": 3,
                },
                "score_list": {
                    "type": "array",
                    "items": score_item,
                    "minItems": 12,
                    "maxItems": 12,
                },
            },
            "required": [
                "stop_distance",
                "path_plan",
                "selected_image",
                "backup_direction_list",
                "score_list",
            ],
            "additionalProperties": False,
        }
        landmark_item = {
            "type": "object",
            "properties": {
                "landmark_name": {"type": "string", "maxLength": 80},
            },
            "required": ["landmark_name"],
            "additionalProperties": False,
        }
        landmark_search = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_id": image_id,
                            "result_list": {
                                "type": "array",
                                "items": landmark_item,
                                "minItems": 0,
                                "maxItems": 12,
                            },
                        },
                        "required": ["image_id", "result_list"],
                        "additionalProperties": False,
                    },
                    "minItems": 12,
                    "maxItems": 12,
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }
        schemas = {
            "spatial_extract_instruction": {
                "type": "object",
                "properties": {
                    "stop_type": {"type": "integer", "enum": [0, 1]},
                    "elevation_decrease": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 0,
                        "maxItems": 20,
                    },
                    "step_actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_name": {"type": "string", "maxLength": 180},
                                "landmarks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "landmark_name": {"type": "string", "maxLength": 80},
                                            "relative_spatial_area": {
                                                "type": "string",
                                                "enum": ["front", "back", "left", "right"],
                                            },
                                        },
                                        "required": ["landmark_name", "relative_spatial_area"],
                                        "additionalProperties": False,
                                    },
                                    "minItems": 0,
                                    "maxItems": 8,
                                },
                            },
                            "required": ["action_name", "landmarks"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 20,
                    },
                },
                "required": ["stop_type", "elevation_decrease", "step_actions"],
                "additionalProperties": False,
            },
            "landmark_search": landmark_search,
            "navigation_alignment": navigation_alignment,
            "stop_decision": {
                "type": "object",
                "properties": {
                    "stop_now": yes_no,
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "brief_reason": short_reason,
                },
                "required": ["stop_now", "confidence", "brief_reason"],
                "additionalProperties": False,
            },
            "bev_align": {
                "type": "object",
                "properties": {
                    "Next_Viewpoint": {"type": "string", "pattern": "^[0-9]+$"},
                    "Reasoning": short_reason,
                },
                "required": ["Next_Viewpoint", "Reasoning"],
                "additionalProperties": False,
            },
            "frontier": {
                "type": "object",
                "properties": {
                    "Selected_Frontier": {"type": "string", "maxLength": 128},
                    "Reasoning": short_reason,
                },
                "required": ["Selected_Frontier", "Reasoning"],
                "additionalProperties": False,
            },
            "synchronize": {
                "type": "object",
                "properties": {
                    "completed_status": yes_no,
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "brief_reason": short_reason,
                },
                "required": ["completed_status", "confidence", "brief_reason"],
                "additionalProperties": False,
            },
            "alternative_stop": {
                "type": "object",
                "properties": {
                    "Destination_Viewpoint": {"type": "string", "pattern": "^(none|[0-9]+)$"},
                    "Reasoning": short_reason,
                },
                "required": ["Destination_Viewpoint", "Reasoning"],
                "additionalProperties": False,
            },
        }
        return schemas[schema_name]

    def get_strict_json_max_tokens(self, schema_name):
        max_by_schema = {
            "spatial_extract_instruction": 1024,
            "landmark_search": 2048,
            "navigation_alignment": 1024,
            "stop_decision": 256,
            "bev_align": 256,
            "frontier": 256,
            "synchronize": 256,
            "alternative_stop": 256,
        }
        min_by_schema = {
            "landmark_search": 1536,
            "navigation_alignment": 768,
        }
        token_budget = min(self.args.max_tokens, max_by_schema.get(schema_name, 512))
        return max(token_budget, min_by_schema.get(schema_name, 1))

    def normalize_strict_json_result(self, result, schema_name):
        if not isinstance(result, dict):
            raise ValueError(f"{schema_name} result must be a JSON object")

        def require_key(key):
            if key not in result:
                raise ValueError(f"{schema_name} missing required field: {key}")
            return result[key]

        def require_string(key, allow_empty=False):
            value = require_key(key)
            if not isinstance(value, str):
                raise ValueError(f"{schema_name} field {key} must be a string")
            value = value.strip()
            if not allow_empty and not value:
                raise ValueError(f"{schema_name} field {key} must not be empty")
            result[key] = value
            return value

        def require_number(key, minimum=None, maximum=None):
            value = require_key(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{schema_name} field {key} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{schema_name} field {key} must be finite")
            if minimum is not None and value < minimum:
                raise ValueError(f"{schema_name} field {key} must be >= {minimum}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{schema_name} field {key} must be <= {maximum}")
            result[key] = value
            return value

        def require_choice(key, choices):
            value = require_string(key)
            if value not in choices:
                raise ValueError(f"{schema_name} field {key} must be one of {sorted(choices)}")
            return value

        if schema_name == "spatial_extract_instruction":
            stop_type = require_key("stop_type")
            try:
                stop_type = int(stop_type)
            except (TypeError, ValueError):
                raise ValueError("spatial_extract_instruction stop_type must be 0 or 1")
            if stop_type not in (0, 1):
                raise ValueError("spatial_extract_instruction stop_type must be 0 or 1")
            result["stop_type"] = stop_type

            elevation_decrease = require_key("elevation_decrease")
            if not isinstance(elevation_decrease, list):
                raise ValueError("spatial_extract_instruction elevation_decrease must be a list")
            result["elevation_decrease"] = [int(item) for item in elevation_decrease]

            step_actions = require_key("step_actions")
            if not isinstance(step_actions, list) or not step_actions:
                raise ValueError("spatial_extract_instruction step_actions must be a non-empty list")
            for action in step_actions:
                if not isinstance(action, dict):
                    raise ValueError("spatial_extract_instruction step_actions items must be objects")
                action_name = action.get("action_name")
                if not isinstance(action_name, str) or not action_name.strip():
                    raise ValueError("spatial_extract_instruction action_name must be a non-empty string")
                action["action_name"] = action_name.strip()
                landmarks = action.get("landmarks")
                if not isinstance(landmarks, list):
                    raise ValueError("spatial_extract_instruction landmarks must be a list")
                for landmark in landmarks:
                    if not isinstance(landmark, dict):
                        raise ValueError("spatial_extract_instruction landmark items must be objects")
                    landmark_name = landmark.get("landmark_name")
                    if not isinstance(landmark_name, str) or not landmark_name.strip():
                        raise ValueError("spatial_extract_instruction landmark_name must be a non-empty string")
                    landmark["landmark_name"] = landmark_name.strip()
                    area = landmark.get("relative_spatial_area")
                    if area not in ("front", "back", "left", "right"):
                        raise ValueError("spatial_extract_instruction relative_spatial_area must be front/back/left/right")
            return result

        if schema_name == "landmark_search":
            raw_results = result.get("results")
            if not isinstance(raw_results, list):
                raise ValueError("landmark_search results must be a list")
            if len(raw_results) != 12:
                raise ValueError(f"landmark_search results must contain 12 items, got {len(raw_results)}")

            results_by_image = {}
            for raw_item in raw_results:
                if not isinstance(raw_item, dict):
                    raise ValueError("landmark_search result items must be objects")
                image_idx = int(raw_item["image_id"])
                if image_idx < 0 or image_idx > 11:
                    raise ValueError(f"landmark_search image_id out of range: {image_idx}")
                if image_idx in results_by_image:
                    raise ValueError(f"landmark_search has duplicate image_id: {image_idx}")

                raw_landmarks = raw_item.get("result_list", [])
                if not isinstance(raw_landmarks, list):
                    raise ValueError(f"landmark_search result_list must be a list for image_id {image_idx}")

                seen_landmarks = set()
                result_list = []
                for raw_landmark in raw_landmarks:
                    if not isinstance(raw_landmark, dict):
                        raise ValueError(f"landmark_search landmarks must be objects for image_id {image_idx}")
                    landmark_name = str(raw_landmark["landmark_name"]).strip()
                    if not landmark_name or landmark_name in seen_landmarks:
                        continue
                    seen_landmarks.add(landmark_name)
                    result_list.append({"landmark_name": landmark_name})

                results_by_image[image_idx] = {
                    "image_id": image_idx,
                    "result_list": result_list,
                }

            missing_images = [image_idx for image_idx in range(12) if image_idx not in results_by_image]
            if missing_images:
                raise ValueError(f"landmark_search missing image_id(s): {missing_images}")

            result["results"] = [results_by_image[image_idx] for image_idx in range(12)]
            return result

        if schema_name == "stop_decision":
            require_choice("stop_now", {"yes", "no"})
            require_number("confidence", 0.0, 1.0)
            require_string("brief_reason")
            return result

        if schema_name == "bev_align":
            next_viewpoint = require_string("Next_Viewpoint")
            if not re.match(r"^[0-9]+$", next_viewpoint):
                raise ValueError("bev_align Next_Viewpoint must contain only digits")
            require_string("Reasoning")
            return result

        if schema_name == "frontier":
            require_string("Selected_Frontier")
            require_string("Reasoning")
            return result

        if schema_name == "synchronize":
            require_choice("completed_status", {"yes", "no"})
            require_number("confidence", 0.0, 1.0)
            require_string("brief_reason")
            return result

        if schema_name == "alternative_stop":
            destination = require_string("Destination_Viewpoint")
            if destination != "none" and not re.match(r"^[0-9]+$", destination):
                raise ValueError("alternative_stop Destination_Viewpoint must be 'none' or digits")
            require_string("Reasoning")
            return result

        if schema_name != "navigation_alignment":
            return result

        selected_image = int(result["selected_image"])
        backup_images = [int(item) for item in result.get("backup_direction_list", [])[:3]]
        if selected_image < 0 or selected_image > 11:
            raise ValueError(f"navigation_alignment selected_image out of range: {selected_image}")
        for backup_image in backup_images:
            if backup_image < 0 or backup_image > 11:
                raise ValueError(f"navigation_alignment backup_direction_list image_id out of range: {backup_image}")
        result["selected_image"] = selected_image
        result["backup_direction_list"] = backup_images

        raw_score_list = result.get("score_list")
        if not isinstance(raw_score_list, list):
            raise ValueError("navigation_alignment score_list must be a list")
        if len(raw_score_list) != 12:
            raise ValueError(f"navigation_alignment score_list must contain 12 items, got {len(raw_score_list)}")

        score_by_image = {}
        for raw_item in raw_score_list:
            if not isinstance(raw_item, dict):
                raise ValueError("navigation_alignment score_list items must be objects")
            image_idx = int(raw_item["image_id"])
            if image_idx < 0 or image_idx > 11:
                raise ValueError(f"navigation_alignment score_list image_id out of range: {image_idx}")
            if image_idx in score_by_image:
                raise ValueError(f"navigation_alignment score_list has duplicate image_id: {image_idx}")
            score = float(raw_item["similarity_score"])
            if not math.isfinite(score):
                raise ValueError(f"navigation_alignment score_list score is not finite for image_id {image_idx}")
            if score < 0.0 or score > 1.0:
                raise ValueError(f"navigation_alignment score_list score out of range for image_id {image_idx}: {score}")
            score_by_image[image_idx] = {
                "image_id": image_idx,
                "similarity_score": score,
            }

        missing_images = [image_idx for image_idx in range(12) if image_idx not in score_by_image]
        if missing_images:
            raise ValueError(f"navigation_alignment score_list missing image_id(s): {missing_images}")

        score_list = [score_by_image[image_idx] for image_idx in range(12)]
        if max(item["similarity_score"] for item in score_list) <= 0.0:
            raise ValueError("navigation_alignment score_list scores are all zero")

        result["score_list"] = score_list
        return result

    def infer_strict_gpt_json(self, prompt_system, prompt_user, image_list, schema_name, context):
        max_attempts = 3
        retry_prompt_user = prompt_user
        last_error = None
        tokens = None
        for attempt in range(max_attempts):
            nav_output, tokens = gpt_infer(
                prompt_system,
                retry_prompt_user,
                image_list,
                self.args.llm,
                self.get_strict_json_max_tokens(schema_name),
                json_schema=self.get_strict_json_schema(schema_name),
            )
            try:
                result = self.load_gpt_json(nav_output, context)
                return self.normalize_strict_json_result(result, schema_name), tokens
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = e
                if attempt == max_attempts - 1:
                    if schema_name == "landmark_search":
                        print(f"[GPT JSON Fallback] {context}: using empty landmark search results after validation failure: {e}", flush=True)
                        return {
                            "results": [
                                {"image_id": image_idx, "result_list": []}
                                for image_idx in range(12)
                            ]
                        }, tokens
                    raise
                print(f"[GPT JSON Validation Retry] {context}: {e}", flush=True)
                retry_prompt_user = prompt_user + "\n\nYour previous JSON was invalid: " + str(e)[:600]
                if schema_name == "navigation_alignment":
                    retry_prompt_user += (
                        "\nReturn one complete JSON object only. For score_list, include exactly 12 objects "
                        "with image_id 0 through 11, no duplicates, and similarity_score numbers from 0 to 1."
                    )
                elif schema_name == "landmark_search":
                    retry_prompt_user += (
                        "\nReturn one complete JSON object only. For results, include exactly 12 objects "
                        "with image_id 0 through 11. Each result_list item must contain only landmark_name."
                    )
                else:
                    retry_prompt_user += "\nReturn one complete JSON object only, matching the required schema."
        raise RuntimeError(f"{context} failed strict JSON validation: {last_error}")

    def reset(self):
        if self.SpatialKnowledgeGraph is not None:
            self.SpatialKnowledgeGraph.clear()
        self.Trajectory = []
        self.intersections = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        self.actionlist = []
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.check_down_elevation=False
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0
        # reset trajectory display bookkeeping
        self.traj_display_id = []
        self.traj_first_visit = {}
        self.floor_ref_z = None
        self.floor_display_counter = 0
        self.floor_changed = False
        self.semantic_check=True
        self.syn_flag=True
        self.uncertain_flag=False
        self.stop_alternative_list=[]
        self.back_to_stop=False
        self.path_to_alt=None
        self.next_viewpoint_to_stop=None

    def spatial_extract_instruction(self, obs):
        # parse the instruction into a sequence of actions and landmarks with spatial area
        prompt_system, prompt_user=self.prompt_manager.make_spatial_extract_instruction_prompt(obs)
        #print('extract instruction prompt:', prompt_system, prompt_user)
        if self.args.response_format == 'json':
                json_output, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    [],
                    "spatial_extract_instruction",
                    "spatial_extract_instruction",
                )
                self.actionlist = json_output['step_actions']
                try:
                    self.stop_type = int(json_output['stop_type'])
                except ValueError:
                    self.stop_type = 0
                # self.stop_type=0 #debug
                # print('stop type:', self.stop_type)
                # check if the last action has no related landmarks, remove the action from the list
                if len(self.actionlist)>0:            
                    last_action_withdot=self.actionlist[-1]['action_name'].lower()
                    last_action_name = re.sub(r'[^\w\s]', '', last_action_withdot)
                    # print('last action name:', last_action_name)
                    if last_action_name=='stop' \
                        or last_action_name=='stop there'\
                        or last_action_name=='wait'\
                        or last_action_name=='wait there'\
                        or last_action_name=='wait here':
                        self.actionlist.pop()
                
                print('Navigation Instruction Steps:', self.actionlist)

                    # # if last landmark is stair, append a stop action with the stair landmark
                    # for item in self.actionlist[-1]['landmarks']:
                    #     if "stair" in item['landmark_name']:
                    #       self.actionlist.append({'action_type':'Stop', 'action_name': 'Stop', 'landmarks': item['landmarks']})
                # print('elevation decrease steps from instruction:', json_output['elevation_decrease'])
                if len(json_output['elevation_decrease'])>0:
                    self.check_down_elevation=True
                    self.down_elevation_steps=json_output['elevation_decrease']

    def build_landmark_dcel_area(self, search_result_list, viewIndex):
        front_landmarks=[]
        back_landmarks=[]
        left_landmarks=[]
        right_landmarks=[]
        viewIndex=viewIndex%12
        # print('Agent heading viewIndex:', viewIndex)

        for image_landmarks in search_result_list:
            relative_index=int(image_landmarks['image_id'])-viewIndex
            
            
            if relative_index<0:
                relative_index=relative_index+12

            if 0<=relative_index<=1 or relative_index==11: # front
                for item in image_landmarks['result_list']:
                    if item['landmark_name'] not in front_landmarks:
                        front_landmarks.append(item['landmark_name'])
            elif 4<=relative_index<=8: # back, wider for depth reasoning
                for item in image_landmarks['result_list']:
                    if item['landmark_name'] not in back_landmarks:
                        back_landmarks.append(item['landmark_name'])
            elif 9<=relative_index <=10: # left
                for item in image_landmarks['result_list']:
                    if item['landmark_name'] not in left_landmarks:
                        left_landmarks.append(item['landmark_name'])
            elif 2<= relative_index <= 3: # right, may add more right images
                for item in image_landmarks['result_list']:
                    if item['landmark_name'] not in right_landmarks:
                        right_landmarks.append(item['landmark_name'])
        landmark_dcel_area={'front':front_landmarks, 'back':back_landmarks, 'left':left_landmarks, 'right':right_landmarks}
        return landmark_dcel_area
    
    # record visual observations in spatial knowledge graph
    def spatial_representation(self, obs, t, env):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)        
        
        if len(obs) > 1:
            print('Error: multiple observations in one step')
        ob = obs[0]
        connect_index=-1  
        path_index=-1     
        print('-----------Move to New Obervation Point-------------')

        # if back_to_stop flag is true, use path_to_alt to select the next node and return
        if self.back_to_stop==True and self.path_to_alt is not None:
            print('Back to stop point, follow the alternative path')
            # if path_to_alt is empty, it means we have arrived at the stop point, set stop flag to true and return
            if len(self.path_to_alt) == 1:
                self.stop_flag=True
                print('Arrived at the alternative stop point, Navigation STOP')
                return 0
            next_node_id=self.path_to_alt[1] # the next step towards the alternative viewpoint
            self.path_to_alt=self.path_to_alt[1:]
            for candidate in ob['candidate']:
                if candidate['viewpointId']==next_node_id:
                    print('Next node on the alternative path:', next_node_id)
                    self.next_viewpoint_to_stop=candidate
                    return 2
            print('Error: next node on the alternative path not found in current candidates')
            self.stop_flag=True
            return 0
        # finish back to stop

        if self.stopping==True: 
            if bool(self.args.stop_after):
                if t >= self.args.stop_after:
                    # if -0.5<self.prev_distance<0.1: # with additional distance estimation of 0
                    #     self.stopping=False
                    #     print('Additional distance estimation invalid, Continue')
                    # else:
                    self.stop_flag=True
                    print('Close Enough to Destination, Navigation STOP')
                    return 0
                else:
                    print('Stop_after not reached, Continue')
                    self.stopping=False
        
            
        # check if the current observation node is intersection with >2 connected observation nodes)
        if t == 0: # no matter start node type, add it as the first path intersection
            self.intersections.append(ob['viewpoint'])

        if len(ob['candidate']) > 2: # intersection
            if t > 0:
                self.intersections.append(ob['viewpoint'])
            connect_index=len(self.intersections) #from 1
            
        elif len(ob['candidate']) == 2: # intermidate node between intersection
            connect_index=0
            
        elif len(ob['candidate']) == 1: # dead end
            if t > 0:
                self.intersections.append(ob['viewpoint']) # dead end is also an path intersection
            connect_index=-1
            
        path_index=len(self.intersections) #from 1
        current_step_landmarks=[]

        j=self.current_instru_step-1
        if t==0: # start node
            j=self.current_instru_step
        # print('[--------Actionlist----------]:', self.actionlist)
        for i in range(j, len(self.actionlist)):  #include current step to last's all landmarks from instruction         
            for landmark in self.actionlist[i]["landmarks"]:
                current_step_landmarks.append(landmark['landmark_name'])
        current_step_landmarks=list(set(current_step_landmarks))
        # print('[-------Current_step---------]:', self.current_instru_step)
        # print('after remove duplicate landmarks,landmarks to check:', current_step_landmarks)
        if t==0: # start node
            orientation=ob['viewIndex']%12
        else:
            orientation=self.current_viewIndex%12
   
       
        final_instru = self.actionlist[len(self.actionlist)-1]['action_name']

        final_instru = final_instru.replace("up the stairs", "up the stairs and stop at the top")
        final_instru = final_instru.replace("down the stairs", "down the stairs and stop at the bottom")
        # print('Final Instruction:', final_instru)
        # print('Full instruction:', full_instru)
        search_result_list=[]
        estimate_distance=-1
        if len(current_step_landmarks) > 0: # if with landmark instructed, search landmarks in the current observation
            prompt_system, prompt_user = self.prompt_manager.make_spatial_search_landmarks_prompt(orientation, final_instru, current_step_landmarks)
            # prompt_system, prompt_user = self.prompt_manager.make_spatial_search_landmarks_prompt(orientation, full_instru, current_step_landmarks)

            # print('search landmarks prompt:', prompt_system, prompt_user)
            image_list = []
            for ix in range(12):
                img_path = os.path.join(env.args.img_root, ob['scan'], ob['viewpoint'], str(ix+12) + '.jpg')
                image_list.append(img_path)               

            if self.args.response_format == 'json':
                json_out, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    image_list,
                    "landmark_search",
                    "initial_decision",
                )
                # estimate_distance=json_out['stop_distance']
                # print('1st Distance Estimation (Meter):', estimate_distance)
                # print('last step distance:', self.stop_distance)
                
                search_result_list=json_out['results']
                # print('search_result_list:', search_result_list)           
        else:
            print('No Related Landmarks for Visual Search')
        

        
        if(t==0): # start node
            self.current_viewIndex=ob['viewIndex']%12
        dcel_area=self.build_landmark_dcel_area(search_result_list, self.current_viewIndex)
       
        tmp_label=self.current_viewIndex
        print('[GPT Visual Observations for Updating SKG]')

        #add candidate nodes and edges to the spatial knowledge graph
        candidate_list=[]
        for candidate in ob['candidate']:           
            self.SpatialKnowledgeGraph.add_node(str(t)+"-"+candidate['viewpointId'],index=candidate['viewpointId'],
                label=candidate['pointId']%12, prev_label=tmp_label, visited=False, intersection_index=-1, path_index=-1,
                position=candidate['position'], heading=candidate['heading'], elevation=candidate['elevation'],
                search_landmarks=[], landmarks_spatial_area=[], action_step=t,candidates=[], backup=[], instru_step=self.current_instru_step,location_estimation=-1)
            candidate_list.append(candidate)
            print('     Add Candidate Node:', candidate['viewpointId'])
            # print('its image direction:', candidate['pointId'])
            
        if ob['viewpoint'] in self.SpatialKnowledgeGraph.nodes:
            if self.SpatialKnowledgeGraph.nodes[ob['viewpoint']].get('prev_label') is not None:
                tmp_label=self.SpatialKnowledgeGraph.nodes[ob['viewpoint']]['prev_label']
        
        self.SpatialKnowledgeGraph.add_node(ob['viewpoint'],index=ob['viewpoint'],
            label=self.current_viewIndex,prev_label=tmp_label, visited=True, intersection_index=connect_index, path_index=path_index,
            position=ob['position'], heading=ob['heading'], elevation=ob['elevation'], 
            search_landmarks=current_step_landmarks, landmarks_spatial_area=search_result_list, 
            dcel_area=dcel_area, action_step=t, candidates=candidate_list, backup=[], instru_step=self.current_instru_step, location_estimation=float(estimate_distance))
       
        self.current_node = self.SpatialKnowledgeGraph.nodes[ob['viewpoint']]
        print('     Add Observation Node:', ob['viewpoint'])  
        print('     Set Node Property of Agent Orientation:', self.current_viewIndex)
        print('     Set Node Property of VLN-DCEL:', dcel_area)
        # print('its networkx id:', t)
        self.Trajectory.append(self.current_node)
        # If agent changed floors (large vertical change), reset traj display bookkeeping
        
        new_state=self.env.env.sims[0].getState()[0]

        try:
            current_world_z = float(new_state.location.z)
        except Exception:
            current_world_z = None
        # print(f"Current world Z: {current_world_z}, Previous floor reference Z: {self.floor_ref_z}")
        if current_world_z is not None:
            if self.floor_ref_z is None:
                self.floor_ref_z = current_world_z
            else:
                if abs(current_world_z - self.floor_ref_z) > self.floor_delta_thresh:
                    print(f"[INFO] SpatialExpert detected floor change: {self.floor_ref_z} -> {current_world_z}. Clearing traj display ids.")
                    # clear display bookkeeping so subsequent appended ids start from 0
                    self.traj_display_id = []
                    self.traj_first_visit = {}
                    # reset per-floor counter so new display ids start at 0
                    self.floor_display_counter = 0
                    # update reference
                    self.floor_ref_z = current_world_z
                    self.floor_changed = True
        
        # maintain traj_display_id: append the first-visit t for this viewpoint
        try:
            vp = ob['viewpoint']
        except Exception:
            vp = None
        if vp is not None:
            if vp in self.traj_first_visit:
                display_t = self.traj_first_visit[vp]
            else:
                # assign a per-floor display id starting from 0
                display_t = int(self.floor_display_counter)
                self.traj_first_visit[vp] = display_t
                self.floor_display_counter += 1
            self.traj_display_id.append(int(display_t))

         # record visited and remaining node locations for BEV mapping/debugging
        try:
            self.record_all_nodes()
        except Exception as e:
            print(f"Warning: record_all_nodes failed: {e}")

        # notify bev app to build bev map

        try:
            is_reverie = getattr(self.args, 'dataset', '').lower() == 'reverie'
        except Exception:
            is_reverie = False
        if t == 0:
            msg = {"type": "reset", "scan": ob['scan'], "viewpoint": ob['viewpoint'], "heading": new_state.heading, "elevation": new_state.elevation, "LX": new_state.location.x, "LY": new_state.location.y, "LZ": new_state.location.z, "is_reverie": is_reverie}
        else:
            msg = {"type": "step", "scan": ob['scan'], "viewpoint": ob['viewpoint'], "heading": new_state.heading, "elevation": new_state.elevation, "LX": new_state.location.x, "LY": new_state.location.y, "LZ": new_state.location.z, "is_reverie": is_reverie}
        print('Requesting BEV build in-process:', msg)
        if self.bev_manager is None:
            raise RuntimeError("BEV manager not initialized")
        result = self.bev_manager.build_with_retry(msg)
        if not result.get("ok"):
            raise RuntimeError(f"bev manager error: {result}")
        print("Received notification of bev map ready")
        return 1 

    def record_all_nodes(self):
        """Write node locations to bevbuilder/bev/node_location.txt.

        The file will contain a JSON object with two lists:
        - "trajectory": list of visited trajectory nodes in visit order, each as
          {"viewpointId":..., "x":..., "y":..., "z":...}
        - "other_nodes": list of SKG nodes whose `index` is not in the
          trajectory, each as {"viewpointId":..., "x":..., "y":..., "z":...}
        """
        out_dir = os.path.join("bevbuilder", "bev")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "node_location.txt")

        # Collect trajectory indices
        traj_indices = []
        traj_list = []
        for node in self.Trajectory:
            try:
                vp_index = node.get('index')
                pos = node.get('position')
                if pos is None and 'position' in node:
                    pos = node['position']
                if vp_index is None:
                    # try viewpointId
                    vp_index = node.get('viewpointId')
                if pos is not None:
                    # normalize position to tuple
                    try:
                        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                    except Exception:
                        # skip malformed
                        continue
                    traj_list.append({"viewpointId": vp_index, "x": x, "y": y, "z": z})
                    traj_indices.append(vp_index)
            except Exception:
                continue

        # assign display_number for trajectory entries (start from 0)
        for i, item in enumerate(traj_list):
            item['display_number'] = int(i)

        other_nodes_list = []
        # Collect only the candidate nodes that are directly connected to the
        # agent's current viewpoint (the last trajectory node). This makes
        # `other_nodes` represent unvisited candidates adjacent to the agent.
        try:
            current_vp = None
            if len(traj_list) > 0:
                current_vp = traj_list[-1].get('viewpointId')

            other_nodes_map = {}
            if current_vp is not None:
                # Find the SKG node corresponding to current_vp
                skg_node = None
                if current_vp in self.SpatialKnowledgeGraph.nodes:
                    skg_node = self.SpatialKnowledgeGraph.nodes[current_vp]
                else:
                    for _nid, _ndata in self.SpatialKnowledgeGraph.nodes(data=True):
                        if _ndata.get('index') == current_vp:
                            skg_node = _ndata
                            break

                if skg_node is not None:
                    cand_items = skg_node.get('candidates', [])
                    for cand in cand_items:
                        try:
                            if not isinstance(cand, dict):
                                continue
                            cvp = cand.get('viewpointId')
                            if cvp is None:
                                continue
                            if cvp in traj_indices:
                                continue

                            # lookup SKG node data for candidate viewpoint to get position
                            pos = None
                            if cvp in self.SpatialKnowledgeGraph.nodes:
                                pos = self.SpatialKnowledgeGraph.nodes[cvp].get('position')
                            else:
                                for _nid, _ndata in self.SpatialKnowledgeGraph.nodes(data=True):
                                    if _ndata.get('index') == cvp:
                                        pos = _ndata.get('position')
                                        break

                            if pos is None:
                                # candidate entry may include position directly
                                pos = cand.get('position')
                            if pos is None:
                                continue
                            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                            if cvp in other_nodes_map:
                                continue
                            other_nodes_map[cvp] = {"viewpointId": cvp, "x": x, "y": y, "z": z}
                        except Exception:
                            continue

            # Preserve insertion order into list
            for vp, entry in other_nodes_map.items():
                other_nodes_list.append(entry)
        except Exception:
            other_nodes_list = []

        # continue display numbering for other_nodes_list starting after trajectory
        start_idx = len(traj_list)
        for i, item in enumerate(other_nodes_list):
            item['display_number'] = int(start_idx + i)

        # continue display numbering for other_nodes_list starting after trajectory
        start_idx = len(traj_list)
        for i, item in enumerate(other_nodes_list):
            item['display_number'] = int(start_idx + i)

        # build mapping: for each trajectory node, list its unvisited candidates
        traj_to_candidates = []
        try:
            # build a quick lookup from other viewpointId -> display_number
            other_vp_to_display = {item['viewpointId']: item['display_number'] for item in other_nodes_list}

            for t_item in traj_list:
                try:
                    t_vp = t_item.get('viewpointId')
                    t_display = t_item.get('display_number')
                    # find the corresponding SKG node data (by graph node key or index)
                    skg_node_data = None
                    if t_vp in self.SpatialKnowledgeGraph.nodes:
                        skg_node_data = self.SpatialKnowledgeGraph.nodes[t_vp]
                    else:
                        # search by ndata.index
                        for _nid, _ndata in self.SpatialKnowledgeGraph.nodes(data=True):
                            if _ndata.get('index') == t_vp:
                                skg_node_data = _ndata
                                break

                    candidate_displays = []
                    candidate_viewpoints = []
                    if skg_node_data is not None:
                        cand_items = skg_node_data.get('candidates')
                        if cand_items and isinstance(cand_items, (list, tuple)):
                            for cand in cand_items:
                                try:
                                    if not isinstance(cand, dict):
                                        continue
                                    cvp = cand.get('viewpointId')
                                    if cvp is None:
                                        continue
                                    if cvp in traj_indices:
                                        continue
                                    # include only if candidate is present in other_nodes_list
                                    if cvp in other_vp_to_display:
                                        candidate_displays.append(other_vp_to_display[cvp])
                                        candidate_viewpoints.append(cvp)
                                except Exception:
                                    continue

                    traj_to_candidates.append({
                        "trajectory_viewpointId": t_vp,
                        "trajectory_display_number": t_display,
                        "candidate_viewpointIds": candidate_viewpoints,
                        "candidate_display_numbers": candidate_displays,
                    })
                except Exception:
                    continue
        except Exception:
            traj_to_candidates = []

        payload = {"trajectory": traj_list, "other_nodes": other_nodes_list, "traj_to_candidates": traj_to_candidates}

        try:
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            # lightweight debug print
            # print(f"Wrote node locations: {len(traj_list)} trajectory nodes, {len(other_nodes_list)} other nodes -> {out_path}")
        except Exception as e:
            print(f"Error writing node locations to {out_path}: {e}")
  
    #     return matched_nodelist
    def GPT_front_landmark_aligned(self, instru_step, ob,t): 
        def sort_candidates_elevation(candidate_list):
            sorted_list = sorted(candidate_list, key=lambda x: x['elevation'])
            return sorted_list
        import ast

        import ast

        def parse_numbers(input_data):
            """Convert a string of numbers or a list into a Python list of integers."""
            
            # If input is already a list, return it directly
            if isinstance(input_data, list):
                return [int(x) for x in input_data]  # Ensure all elements are integers
            
            # Convert to string and remove spaces
            input_str = str(input_data).strip()

            try:
                # If input looks like a list (contains [ and ]), use ast.literal_eval
                if input_str.startswith("[") and input_str.endswith("]"):
                    return ast.literal_eval(input_str)
                else:
                    # Otherwise, assume comma-separated numbers and split manually
                    return [int(x.strip()) for x in input_str.split(",")]
            
            except Exception as e:
                print(f"Error parsing input: {e}")
                return []   # Return empty



    
        reason_result=None       
        matched_nodelist=[]
        current_action_name=self.actionlist[instru_step]['action_name']

       
        self.extracted_instruction=ob['instruction']
    #  parse the instruction into a sequence of actions and landmarks with spatial area
        orientation=self.current_viewIndex
        arrival=(self.current_viewIndex+6)%12
        navigable_images=[]
        navigable_images_origin=[]
        navigable_candidates=[]
        for k in range(len(ob['candidate'])):
            candidate = ob['candidate'][k]
            # print('candidate position')
            navigable_images_origin.append(candidate['pointId']%12)
            navigable_candidates.append(candidate)
        navigable_images = list(dict.fromkeys(navigable_images_origin)) # remove duplicates
        # print('navigable_images:', navigable_images)

        if instru_step==len(self.actionlist)-1: # last action
            current_action_name = current_action_name.replace("up the stairs", "up the stairs and stop at the top")
            current_action_name = current_action_name.replace("down the stairs", "down the stairs and stop at the bottom")
            # print('last action name:', current_action_name)

       
        if (instru_step==len(self.actionlist)-1): # last action
            prompt_system, prompt_user=self.prompt_manager.make_gpt_stop_distance_reason_prompt(ob, current_action_name, navigable_images)
        else:
            prompt_system, prompt_user=self.prompt_manager.make_gpt_breadth_distance_reason_prompt(ob, self.extracted_instruction,orientation, navigable_images, current_action_name)


        
        image_list = []
        temp_list=[]
        for ix in range(12):
            img_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(ix+12) + '.jpg')
            image_list.append(img_path) 
   
        if self.args.response_format == 'json':
            reason_result, tokens = self.infer_strict_gpt_json(
                prompt_system,
                prompt_user,
                image_list,
                "navigation_alignment",
                "GPT_front_landmark_aligned",
            )
        # print('reason_result:', reason_result)  
        down_reason_result=None
        current_landmark_name=None
        if self.check_down_elevation==True:
            int_down_steps=[int(item)-1 for item in self.down_elevation_steps]
            # print('down elevation steps:', int_down_steps)
            int_down_steps = sorted(int_down_steps)
            if int_down_steps[0] > 0:
                int_down_steps.insert(0, int_down_steps[0] - 1)
            # print('sorted down elevation steps:', int_down_steps)
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            image_list.clear()
            temp_list.clear()
            for ix in range(12):
                img_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(ix) + '.jpg')
                image_list.append(img_path)
           
            if self.args.response_format == 'json':
                down_reason_result, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    image_list,
                    "navigation_alignment",
                    "GPT_front_landmark_aligned_down",
                )

        # print('Agent Orientation:', self.current_viewIndex)
        landmarks=self.actionlist[instru_step]['landmarks']
        if len(landmarks)>0:
            current_landmark_name = landmarks[0]['landmark_name']
        print('[GPT Spatial Alignment Inference]')
        print('      Landmark at Current Step:', current_landmark_name)
        print('      Execute Movement Action:', current_action_name)
        # print('      Navigable_images:', navigable_images)
        print('      Selected Direction:', reason_result['selected_image'])
        print('      Alternative Direction:', reason_result['backup_direction_list'])
        print('      Subsequent Path Plan:', reason_result['path_plan'])      
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            print('[GPT] Down-Elevation Spatial Alignment Inference-Selected Direction:', down_reason_result['selected_image'])
        matched_nodelist=[]
        down_matched_nodelist=[]
     
        down_highest_candidate=None
        highest_candidate=None
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):     
            down_score_list=down_reason_result['score_list']
            down_navigable_scores=[]
         
            # print('navigable_images_origin:', navigable_images_origin)
            for i in range(len(navigable_images_origin)):
                angle=abs(int(down_reason_result['selected_image'])-int(navigable_images_origin[i]))
                gap=min(angle, 12-angle)
                int_navigation=int(navigable_images_origin[i])
                score=float(down_score_list[int_navigation]['similarity_score'])
                elevation=navigable_candidates[i]['elevation']
                down_navigable_scores.append([navigable_candidates[i], gap, score, elevation])
             
            down_navigable_scores=sorted(down_navigable_scores, key=lambda x: (x[1], -x[2], x[3]))            
           
            traj_nodes=[]
            reversed_list = list(reversed(self.Trajectory))
            
            for node in reversed_list:
                traj_nodes.append(node['index'])            
            # print('traj_nodes:', traj_nodes)
            down_not_in_traj = []
            down_in_traj = []
            for item in down_navigable_scores:
                candidate=item[0]
                if candidate['viewpointId'] in traj_nodes:
                    down_in_traj.append(item)
                else:
                    down_not_in_traj.append(item)

            down_navigable_scores.clear()
            down_navigable_scores=down_not_in_traj+down_in_traj
            # a2 = " ".join(item[0]['viewpointId'] for item in down_navigable_scores)
            # print('after down check visited nodes navigable_scores:', a2)
            down_highest_candidate=down_navigable_scores[0][0]           
            down_matched_nodelist.append(down_highest_candidate)
 
        # set backup node for reflect
        backup_list=reason_result['backup_direction_list']
        # print('gpt return backup list:', backup_list)
        
        backup_list=parse_numbers(backup_list)
        # print('after processed backup list:', backup_list)


        # direction_difflist=[]
        # backup_index=-1
        backup_nodelist=[]     
        # sort candidates by gap increasing, for the samp gap, sort by score decreasing
        score_list=reason_result['score_list']
        navigable_scores=[]
        int_backup_list=[int(item) for item in backup_list]
        # print('navigable_images_origin:', navigable_images_origin)
        for i in range(len(navigable_images_origin)):
            angle=abs(int(reason_result['selected_image'])-int(navigable_images_origin[i]))
            gap=min(angle, 12-angle)
            score=float(score_list[navigable_images_origin[i]]['similarity_score'])
            navigable_scores.append([navigable_candidates[i], gap, score])
            # print('append i',i)
            # print('append navigable_candidates:', navigable_candidates[i])
            int_navigation=int(navigable_images_origin[i])

            if int_navigation in int_backup_list:#remember alternative candidates
                # print('int_navigation:', int_navigation)
                backup_nodelist.append(navigable_candidates[i])
        
       
        navigable_scores=sorted(navigable_scores, key=lambda x: (x[1], -x[2]))
        a1 = " ".join(str(item[1]) for item in navigable_scores)
        # print('sort gap:', a1)
        a2 = " ".join(str(item[2]) for item in navigable_scores)
        # print('sort score:', a2)
        traj_nodes=[]
        reversed_list = list(reversed(self.Trajectory))
        
        for node in reversed_list:
            traj_nodes.append(node['index'])        
        # print('traj_nodes:', traj_nodes)
        not_in_traj = []
        in_traj = []
        for item in navigable_scores:
            candidate=item[0]
            if candidate['viewpointId'] in traj_nodes:
                in_traj.append(item)
            else:
                not_in_traj.append(item)
        if self.stopping==False:
            navigable_scores.clear()
            navigable_scores=not_in_traj+in_traj
 
        highest_candidate=navigable_scores[0][0]
        
        matched_nodelist.append(highest_candidate)
 
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            matched_nodelist=down_matched_nodelist+matched_nodelist
        
        candidate=None
        cirlce_flag=False
        reflect_result = None
        # exclude selected candidate from backup list
        if len(matched_nodelist)>0:
            candidate=matched_nodelist[0]

            # Reverie-specific perform frontier exploration.
            try:
                is_reverie = getattr(self.args, 'dataset', '').lower() == 'reverie'
            except Exception:
                is_reverie = False
            
            if is_reverie == True and t < self.args.stop_after:
                print('[Reverie instruction] initiates bev align')
                # reflect_result = None  # Reverie-specific: skip reflect and force frontier
                # self.frontier_flag=True
                bev_result=self.bev_align(ob)
                if bev_result is not None:
                    matched_nodelist.clear()
                    matched_nodelist.append(bev_result)   
                    print('Bev Align Exploration Result:', bev_result['viewpointId']) 
            # end reverie-specific frontier exploration

            for idx, node in enumerate(reversed(self.Trajectory)):
                if idx == 0:
                    continue
                if self.current_node['index'] == node['index']:
                    cirlce_flag=True
                    break
            
            if cirlce_flag==True:
                self.num_backtrack+=1
                print('[Backtrack Planning Initiated]')                       
                if candidate['viewpointId'] in traj_nodes:
                    # print('[Backtrack Plan due to Circle Found]')                       
                    # go to reflect for out of circle
                    reflect_result=self.reflect(ob)
                    if reflect_result is not None:
                        matched_nodelist.clear()
                        matched_nodelist.append(reflect_result)
                    else:
                        print('Frontier Inference due to no alternative node')
                        if self.frontier_flag==True:
                            frontier_result=self.frontier(ob)
                            if frontier_result is not None:
                                matched_nodelist.clear()
                                matched_nodelist.append(frontier_result)            
            
                # else:   
                #     print('Revisited an Accessed Node, Self-Correct to Continue')

            selected_node=matched_nodelist[0]
            if selected_node in backup_nodelist:
                # print('remove selected node from backup:', selected_node['viewpointId'])
                backup_nodelist.remove(selected_node)

        if len(backup_nodelist)>0: 
            # print('add backup_nodelist:', backup_nodelist)
            self.current_node['backup']=backup_nodelist
       

        possible_stop=False
        # if t>=len(self.actionlist)-1:  # after number of actions
        if bool(self.args.stop_after):
            if t >= self.args.stop_after:
                possible_stop=True
            else:
                print('Stop_after not reached, Continue')

   
         
        try:
            self.stop_distance = float(reason_result['stop_distance'])
        except ValueError:
            print("Error: stop_distance is not a valid float value:", reason_result['stop_distance'])
            self.stop_distance = -1.0

     
        # estimated_distance=self.current_node['location_estimation']
        # print('Additional Estimated Distance (Meter):', estimated_distance)
        if float(self.stop_distance)<-0.5:
            print('Destination Invisible at Distance Estimation')
        else: 
            print('Destination Seen at Distance Estimation (Meter):', self.stop_distance)

        # save fpv for gif
        # dest_heading_index=int(reason_result['selected_image'])%12
        # dest_image_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(dest_heading_index+12) + '.jpg')
        # # copy dest_image_path to current folder with name "fpv.jpg"
        # try:
        #     shutil.copy(dest_image_path, "fpv.jpg")
        # except Exception as e:
        #     print(f"Error copying FPV image for stop decision: {e}")
      
        # endsave

        # if self.stop_distance > -0.5 and possible_stop==True:
        # if self.stop_distance > -0.5 and possible_stop==True and instru_step>=len(self.actionlist)-1: # last action
        if possible_stop==True and instru_step>=len(self.actionlist)-2:
            # dest_heading_index=int(reason_result['selected_image'])%12
                   
            depth=-1
           # use bev for stop decision if destination visible
            # draw selected node on bev map
            # if len(matched_nodelist)>0:
            #     next_node = matched_nodelist[0]
            #     try:
            #         node_location_path = os.path.join("bevbuilder", "bev", "node_location.txt")
            #         node_payload = {}
            #         if os.path.exists(node_location_path):
            #             with open(node_location_path, 'r') as f:
            #                 node_payload = json.load(f)
            #         if not isinstance(node_payload, dict):
            #             node_payload = {}
            #         node_payload["next_node_viewpointId"] = next_node.get("viewpointId")
            #         with open(node_location_path, 'w') as f:
            #             json.dump(node_payload, f, indent=2)
            #     except Exception as e:
            #         print(f"Error writing next_node_viewpointId to node_location.txt: {e}")

            #     try:
            #         if self.bev_manager is not None:
            #             draw_msg = {
            #                 "scan": ob.get("scan"),
            #                 "bev_save_path": "bevbuilder/bev/map.png",
            #                 "cs": 0.01,
            #                 "gs": 3000,
            #             }
            #             draw_result = self.bev_manager.draw_next_node_with_retry(draw_msg)
            #             if not draw_result.get("ok"):
            #                 print(f"Error notifying BEV drawNextNode: {draw_result}")
            #     except Exception as e:
            #         print(f"Error notifying BEV drawNextNode: {e}")
                 
                


            traj_string=""
            for display_id in self.traj_display_id:
                traj_string+=str(display_id)+"->"
            traj_string=traj_string[:-2] # remove the last ->
            # tmp_instr=current_action_name # last action only
            tmp_instr=self.extracted_instruction # full instruction for better reasoning of stop now or not
            
            # tmp_instr = tmp_instr.replace("up the stairs", "up the stairs and stop at the top")
            # tmp_instr = tmp_instr.replace("down the stairs", "down the stairs and stop at the bottom")
            stop_stair_end=False
            if re.search(r"\bup the stairs(?=[\s\.\!\?,;:]*$)", tmp_instr, re.IGNORECASE):
                if "stop at the top" not in tmp_instr:
                    tmp_instr = re.sub(
                        r"\bup the stairs(?=[\s\.\!\?,;:]*$)",
                        "up the stairs and stop at the top",
                        tmp_instr,
                        flags=re.IGNORECASE
                    )
                    stop_stair_end=True


            if re.search(r"\bdown the stairs(?=[\s\.\!\?,;:]*$)", tmp_instr, re.IGNORECASE):
                if "stop at the bottom" not in tmp_instr:
                    tmp_instr = re.sub(
                        r"\bdown the stairs(?=[\s\.\!\?,;:]*$)",
                        "down the stairs and stop at the bottom",
                        tmp_instr,
                        flags=re.IGNORECASE
                    )
                    stop_stair_end=True

            # prompt_system, prompt_user=self.prompt_manager.make_stop_bybev_prompt(tmp_instr, traj_string)        
            # if self.semantic_check==True:
            #     prompt_system, prompt_user=self.prompt_manager.make_stop_bybev2_prompt(tmp_instr, traj_string)        
            #     print('Using Semantic Check for Stop Decision with BEV Map')
            # if self.stop_type==0: # object related stop, use local bev
            #     prompt_system, prompt_user=self.prompt_manager.make_stop_bevobj_prompt(tmp_instr, traj_string)        
            # else:
            prompt_system, prompt_user=self.prompt_manager.make_stop_bev_approach_prompt(tmp_instr, traj_string)        
            # print('stop_type:', self.stop_type)
            # print('stop prompt:', prompt_system, prompt_user)        
            image_list = []
            image_list.clear()
            img_path = os.path.join("bevbuilder/bev/map_zoom_color.png")

            # print('bev map path 1 for stop:', img_path)
            #verify the image path exists 
            if not os.path.exists(img_path):
                print(f"Error: BEV map image file not found at {img_path}")
                
            image_list.append(img_path)
            if self.semantic_check==True:
                img_path = os.path.join("bevbuilder/bev/map_zoom_semantic_legend.png")
                  # print('bev map path 2 for stop:', img_path)
                #verify the image path exists 
                if not os.path.exists(img_path):
                    print(f"Error: BEV map image file not found at {img_path}")
                    
                image_list.append(img_path)
            # print('number of images for stop decision:', len(image_list))
            bev_distance=100.0
            if self.args.response_format == 'json':
                    stop_result, tokens = self.infer_strict_gpt_json(
                        prompt_system,
                        prompt_user,
                        image_list,
                        "stop_decision",
                        "stop_decision",
                    )
                    # print('stop_result:', stop_result)
                    # try:
                    #     bev_distance = float(stop_result['estimated_distance'])
                    # except ValueError:
                    #     print("Error: bev estimated_distance is not a valid float value:", stop_result['estimated_distance'])
                    print('[GPT Stop Inferenece by VLN-BEV]')
                    # print('      Identified:'+ stop_result['identified'])
                    # print('      Estimated_distance:', stop_result['estimated_distance'])
                    print('      Stop_Now:'+ stop_result['stop_now'])
                    print("      Confidence: " + str(stop_result["confidence"]))
                    print('      Brief Reason:'+stop_result['brief_reason']) 

            
            # if is_reverie is False:
            #     if stop_result['destination_inside_circle']=='yes':
            #         if self.stop_distance < -0.5: # bev for topology-based stop 
            #             self.stop_distance=2.0
            #     else:
            #         self.stop_distance=4.0 # override stop distance to a far range
            # else:
            #     if self.stop_distance < -0.5: # bev for topology-based stop 
            #         self.stop_distance =4.0 # 
            
            # if stop_result['destination_inside_circle']=='yes':
            #     if self.uncertain_flag==True: 
            #         print('Uncertain stage: BEV inside again')
            #     if self.stop_distance > -0.5 and self.stop_distance < 3: # FPV and BEV to stop
            #         self.stop_distance=2.0 # to stop  
            #         print('Both Stop')
            #     else: # BEV only, to uncertain
            #         self.uncertain_flag=True
            #         # self.stop_distance=2.0 # to stop  
            #         print('BEV Stop Only - Enter Uncertain Stage')
            # else: 
            #     if self.stop_distance > -0.5 and self.stop_distance < 3: # FPV only, to stop
            #         self.stop_distance=2.0 
            #         print('FPV Stop Only')
            #     if self.stop_distance < -0.5 or self.stop_distance > 3: # neither, no stop
            #         self.stop_distance=4.0 
            #         print('Neither Stop')

            # if self.stop_distance > -0.5: # FPV indicates close to destination
               
            print('Stop FPV-Estimated Distance:', self.stop_distance) 
            bev_stop=False
            fpv_stop=False
            if stop_result['stop_now']=='yes':
                bev_stop=True
            if self.stop_distance > -0.5 and self.stop_distance < 2.6:
                fpv_stop=True

            
            if bev_stop==True and fpv_stop==True:   
                self.stopping=True
                    
            if bev_stop==True or fpv_stop==True:
                if all(
                    item[0] != self.current_node['index']
                    for item in self.stop_alternative_list
                ):
                    #append current viewpoint, current height, and stop_distance estimation together for better analysis of stop decision
                    self.stop_alternative_list.append((self.current_node['index'], self.floor_ref_z, self.stop_distance))
                    print('Add current viewpoint to stop_alternative_list:', self.current_node['index'])  


            if self.stop_type==1: # if stop on a stair step, fpv is not reliable,use bev only for stop decision
                if bev_stop==True:
                    self.stopping=True
                    print('Stair step destination, Confirming Stop')

            if stop_stair_end==True:
                if self.floor_changed==False: # override the case of stopping early if no floor change detected
                    self.stopping=False
                    self.stop_flag=False
                    # print('No Floor Change Detected, Continue Moving to Search for Stair End')
                else: # agent changed floor
                    self.stop_flag=True
                    print('Stair end destination, Confirming Stop')

           #use groundingdino and depth for stop decision
            # if self.stop_type==0:  #if object related stop, use depth
            #     dest_image_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(dest_heading_index+12) + '.jpg')
            #     roi=None

            #     if current_landmark_name is not None:
            #         roi=self.get_ROI_object_depth(dest_image_path, current_landmark_name)

            #     if roi is not None:
            #         print("Top ROI Box:", roi)
            #     else:
            #         print("No object detected.")

                

            #     #get depth image if detected
            #     if roi is not None:
            #         depth_img = self.get_depth_at_discrete_view(dest_heading_index) 
            #         # use median depth value of the image
            #         x_min, y_min, x_max, y_max = map(int, roi)
            #         roi_depth = depth_img[y_min:y_max, x_min:x_max]
            #         valid_depths = roi_depth[roi_depth > 0]

            #         if valid_depths.size == 0:
            #             print("No valid depth in ROI.")
            #         else:
            #             depth = np.median(valid_depths)/4000.0
            #             if np.isnan(depth) or depth < 0.1 or depth > 10.0:
            #                 depth = -1
            
            #             print(f"Estimated distance to object: {depth:.2f} meters")
            # using depth value if available
            # if depth != -1:   #accurate depth value for stop now
            #     print('!----Using Object Depth:', depth)
            #     if depth < 4.5:
            #         self.stopping=True
            # else:    
            #     if float(self.stop_distance)<2.1 and float(self.stop_distance)>-0.5: # stop distance estimation is valid and close
            #         self.stopping=True
 
    

        if self.stopping==True:
            print('Closing to Destination, Agent is Stopping...')  


        return matched_nodelist
    

    def get_ROI_object_depth(self, image_path, object):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from PIL import Image, ImageDraw


        # image_path = "rgb.png"  # Path to your image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        else:
            print(f"[INFO] Successfully opened image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        text = object  # Text prompt for the model

        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        inputs = processor(images=image, text=text, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]
        image_width, image_height = image.size
        print(f"Image size: {image_width} x {image_height}")
        print(f"Detected {len(results['scores'])} objects:", text)

    
        
        scores = results["scores"]
        boxes = results["boxes"]

        if len(scores) == 0:
            return None 

        top_idx = scores.argmax()
        box = boxes[top_idx].tolist()

        # Clamp + round
        x_min = max(0, min(box[0], image_width))
        y_min = max(0, min(box[1], image_height))
        x_max = max(0, min(box[2], image_width))
        y_max = max(0, min(box[3], image_height))
        clamped_box = [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
        print(f"Detected {object} with confidence {round(scores[top_idx].item(), 2)} at location {box}")
    

        return clamped_box

    def get_depth_at_discrete_view(self, heading_index: int, elevation_index: int = 0):
        sim=self.env.env.sims[0]
        # save current state
        
        state = sim.getState()[0]

        # state = sim.getState()[0]
        scan_id = state.scanId
        viewpoint_id = state.location.viewpointId
        current_heading = state.heading
        current_elevation = state.elevation

        # set discrete heading and elevation
        heading_rad = heading_index * math.radians(30)
        elevation_rad = elevation_index * math.radians(30)
        sim.newEpisode([scan_id], [viewpoint_id], [heading_rad], [elevation_rad])
        new_state = sim.getState()[0]

        # get depth image
        depth = np.array(new_state.depth, copy=False)

        # back to original state
        sim.newEpisode([scan_id], [viewpoint_id], [current_heading], [current_elevation])
        return depth

    def bev_align(self, ob): 
        selected_node=None
        selected_node_number=-1
        traj_string=""
        for display_id in self.traj_display_id:
            traj_string+=str(display_id)+"->"
        traj_string=traj_string[:-2] # remove the last ->
        # img_path = os.path.join("bevbuilder/bev/map_candidates_color.png")
        img_path = os.path.join("bevbuilder/bev/map_candidates_zoom_color.png")

        # print('bev map path for backtrack:', img_path)
        #verify the image path exists 
        if not os.path.exists(img_path):
            print(f"Error: BEV map image file not found at {img_path}")

        image_list = []
        image_list.clear()    
        image_list.append(img_path)

        prompt_system, prompt_user=self.prompt_manager.make_bev_align_prompt(self.extracted_instruction, traj_string)       
        # print('bev align prompt:', prompt_system, prompt_user)        
        if self.args.response_format == 'json':
                bev_result, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    image_list,
                    "bev_align",
                    "stop_bev_approach",
                )
                # print('bev_result:', bev_result)
                print('[GPT Bev Align Inferenece]')
                print('      Next Viewpoint:'+ bev_result['Next_Viewpoint'])
                print('      Reasoning:'+bev_result['Reasoning'])
                selected_node_number=bev_result['Next_Viewpoint']
                # based on the node_location.txt, get the viewpointId with display_number equal to selected_node_number
                selected_viewpointId=None
                node_location_path = os.path.join("bevbuilder", "bev", "node_location.txt")
                if os.path.exists(node_location_path):
                    with open(node_location_path, 'r') as f:
                        node_locations = json.load(f)
                        all_nodes = node_locations.get("trajectory", []) + node_locations.get("other_nodes", [])
                        for node in all_nodes:  
                            if node.get('display_number') == int(selected_node_number):
                                selected_viewpointId = node.get('viewpointId')
                                break
                else:
                    print(f"Error: Node location file not found at {node_location_path}")   
                # retrieve the candidate with viewpointId equal to selected_viewpointId
                if selected_viewpointId is not None:
                    for candidate in ob['candidate']:
                        if candidate['viewpointId'] == selected_viewpointId:
                            selected_node = candidate
                            break
                else:
                    print(f"Error: No node found with display_number {selected_node_number} in node_location.txt")
        return selected_node  

    def frontier(self, ob): # frontier inference
        selected_node=None
        img_path = os.path.join("bevbuilder/bev/map_color.png")
        print('bev map path for backtrack:', img_path)
        #verify the image path exists 
        if not os.path.exists(img_path):
            print(f"Error: BEV map image file not found at {img_path}")

        image_list = []
        image_list.clear()    
        image_list.append(img_path)

        prompt_system, prompt_user=self.prompt_manager.make_frontier_prompt(self.current_instru_step, self.current_node, self.SpatialKnowledgeGraph)         
        print('backtrack prompt:', prompt_system, prompt_user)        
        if self.args.response_format == 'json':
                frontier_result, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    image_list,
                    "frontier",
                    "frontier",
                )
                print('frontier_result:', frontier_result)
                print('[GPT Frontier Inferenece]')
                print('      Selected Frontier:'+ frontier_result['Selected_Frontier'])
                print('      Reasoning:'+frontier_result['Reasoning'])
                selected_node=frontier_result['Selected_Frontier']
        return selected_node      
    
    def reflect(self, ob):  #decide exploit or explore
        self.num_reflect+=1
        reversed_list = list(reversed(self.Trajectory))
        traj_nodes=[]        
        for node in reversed_list:
            traj_nodes.append(node['index'])
        selected_node=None
        prev_candidate=None
        prev_node=None
        # print('     Checking Alternative Nodes From SKG Current Node..')
        for idx, node in enumerate(reversed(self.Trajectory)):
            if idx == 0:
                continue
            if self.current_node['index'] == node['index']: 
                # check backup memory for exploit       
                for j in range(len(self.current_node['backup'])):
                    backup_candidate=self.current_node['backup'][j]
                    if  backup_candidate in traj_nodes: # exclude the visited nodes
                        continue
                    else:
                        selected_node=backup_candidate
                        print('     Retrieve Alternative Node From SKG:', selected_node['viewpointId'])
                # check previous node
                if idx<len(self.Trajectory)-1:
                    prev_node=list(reversed(self.Trajectory))[idx+1]
                # check unvisited candidates for explore
                if selected_node is None:            
                    for i in range(len(ob['candidate'])):
                        candidate = ob['candidate'][i]
                        if candidate['viewpointId'] in traj_nodes: # exclude the visited nodes
                            if prev_node is not None:
                                if candidate['viewpointId'] == prev_node['index']:
                                    prev_candidate=candidate
                            continue     
                        else:
                            selected_node=candidate               
        # check previous node    
        if selected_node is None:  
            print('Reflect Checked Current Node, Explore to Previous Node')
            selected_node=prev_candidate
        return selected_node

    def breadth_reasoning(self, ob, t):
        selected_node = None            
        matched_nodelist=self.GPT_front_landmark_aligned(self.current_instru_step, ob,t)
        if len(matched_nodelist)>0:
            selected_node = matched_nodelist[0] 
            # step increase if found target
            if self.current_instru_step < len(self.actionlist)-1:
                if self.current_node['intersection_index'] > 0 or t==0: # intersection or start node
                    self.current_instru_step+=1
                    # print('forward step continue+1:', self.current_instru_step)         
        return selected_node
    
    def synchronize_reasoning(self, ob, t):    
        def convert_dcel_area_to_text(dcel_area: dict) -> str:
            
            directions_text = []
            # For each direction (front/back/left/right), list the items found
            for direction in ['front', 'back', 'left', 'right']:
                items = dcel_area[direction]
                if items:
                    items_str = ', '.join(items)
                    directions_text.append(f"{direction} area: {items_str}")
                else:
                    directions_text.append(f"{direction} area: (none)")
                    
            # Join them together into one multiline string
            return " ; ".join(directions_text)

        step_str = ""
        for i in range(len(self.actionlist)):
            action = self.actionlist[i]
            step_str += f"""{i+1}.{action['action_name']};"""

        if self.current_instru_step == 0:
            k=0
        else:
            k=self.current_instru_step-1

        prev_step_str = self.actionlist[k]['action_name']
        
        if t==0:
            return True
        
        # if self.last_action==True:
        #     return True

     #dcel method   
        # description_text = convert_dcel_area_to_text(self.current_node['dcel_area'])
        # prompt_system, prompt_user=self.prompt_manager.make_synchronize_bydcel_prompt(step_str, prev_step_str, description_text)         
        # # print('synchronize prompt:', prompt_system, prompt_user)        
        # if self.args.response_format == 'json':
        #         nav_output, tokens = gpt_infer(prompt_system, prompt_user, [],
        #                                     self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
        #         syn_result = json.loads(nav_output)
        #         print('[GPT Temporal Synchronize Inferenece by VLN-DCEL]')
        #         print('      Previous Step Completion:'+ syn_result['completed_status'])
        #         print('      Brief Reason:'+syn_result['brief_reason'])
    #end dcel method
    #bev method
        #traj_string is a string for LLM to understand the trajectory, which is the sequence of display id in traj_display_id
        traj_string=""
        for display_id in self.traj_display_id:
            traj_string+=str(display_id)+"->"
        traj_string=traj_string[:-2] # remove the last ->
        prompt_system, prompt_user=self.prompt_manager.make_synchronize_bybev_prompt(step_str, prev_step_str, traj_string)        
        # print('synchronize prompt:', prompt_system, prompt_user)        
        image_list = []
        image_list.clear()
        img_path = os.path.join("bevbuilder/bev/map_color.png")
        # print('bev map path for synchronization:', img_path)
        #verify the image path exists 
        if not os.path.exists(img_path):
            print(f"Error: BEV map image file not found at {img_path}")
            
        image_list.append(img_path)
        if self.args.response_format == 'json':
                syn_result, tokens = self.infer_strict_gpt_json(
                    prompt_system,
                    prompt_user,
                    image_list,
                    "synchronize",
                    "synchronize",
                )
                print('[GPT Temporal Synchronize Inferenece by VLN-BEV]')
                print('      Previous Instruction:'+ prev_step_str)
                print('      Previous Step Completion:'+ syn_result['completed_status'])
                print("      Confidence: " + str(syn_result["confidence"]))
                print('      Brief Reason:'+syn_result['brief_reason'])
    #end bev method
        if t>0:
            if syn_result['completed_status']=='no':
                landmarks=self.actionlist[self.current_instru_step]['landmarks']
                current_landmark=None
                prev_landmark=None
                if len(landmarks)>0:
                    current_landmark=landmarks[0]['landmark_name']
                landmarks=self.actionlist[self.current_instru_step-1]['landmarks']
                if len(landmarks)>0:
                    prev_landmark=landmarks[0]['landmark_name']
                if current_landmark==prev_landmark and current_landmark is not None:
                    print('Consecutive Steps with the Same Landmark:', current_landmark)
                else:
                    if self.stopping==False:
                        self.current_instru_step-=1    
            # else:
            #     if self.current_instru_step == len(self.actionlist)-1:
            #         # print('Last Step Reached, No Need to Synchronize Any More')     
            #         self.last_action=True
        return True   
    
    
    def continue_straight(self, ob, t):  
        next_node = None
        if (len(ob['candidate']) == 1): # dead-end node
            print('Dead-end Node to the Only Backward Node')
            return ob['candidate'][0]
        
        if len(ob['candidate']) == 2 and t > 0: # intermidate node 
            for candidate in ob['candidate']:
                if candidate['viewpointId'] != self.Trajectory[-2]['index']: #not previous node
                    next_node = candidate
                    break
            print('Intermidate Node to the Only Foward Node')
            return next_node
        
        # select the node closest to the base heading, i.e minimum candidate heading         
        if len(ob['candidate']) > 2 or t == 0: # start node (no-align landmark)
            min_angle=3.141592
            num_candidates=len(ob['candidate'])
            if t > 0: # not start node, previous node out
                num_candidates=num_candidates-1
            # print('straight candidates has:', num_candidates)
            next_node = None
            current_heading=(self.current_viewIndex+0.5)*math.radians(30)
            for candidate in ob['candidate']:   
                # print('candidate absolute heading:', candidate['absolute_heading'])
                gap_angle=abs(candidate['absolute_heading']-current_heading)
                if gap_angle > 3.141592:
                    gap_angle=2*3.141592-gap_angle
                if gap_angle < min_angle:
                    min_angle=gap_angle
                    next_node = candidate
            print('Continue Straight to Node:', next_node)
            # print('its gap angle:', min_angle)            

            return next_node
        print('Error in Continue Straight: No Selected Node')    
        return -1    
    
    # spatial reasoning for the current observation with instruction
    def spatial_reasoning(self, obs, t):
        ob = obs[0]
        selected_node = None       
            
        if t==0: #start node
            # print('Spatial Reasoning On Beginning Node')
            self.extracted_instruction=ob['instruction']
            # if self.depth_reasoning(ob,t)==True: # check progress of instruction
            self.synchronize_reasoning(ob, t)
            selected_node=self.breadth_reasoning(ob, t)
            if selected_node == None:
                print('Alignment: No Node Met Spatial Condition, Go Straight')
                selected_node=self.continue_straight(ob, t) # cannot exploit instruction, go to reflect directly?
                self.current_instru_step+=1
  
        elif self.current_node['intersection_index'] > 0:   # intersection and not the start node
            # print('Spatial Reasoning On Intersection Node')
            # if self.depth_reasoning(ob,t)==True: # check progress of instruction
            if self.synchronize_reasoning(ob, t)==True:
                selected_node=self.breadth_reasoning(ob, t)
                if selected_node is None: # similarity not met, possibility is low
                    print('Alignment: No Node Met Spatial Condition, Go Straight')
                    #selected_node=self.reflect(ob)
                    selected_node=self.continue_straight(ob, t)                    
    

        elif self.current_node['intersection_index'] == 0: # intermidate node between intersection   
            # print('Spatial Reasoning On Intermidate Node')
            selected_node=self.breadth_reasoning(ob, t)
            if selected_node == None:
                print('Alignment: No Node Met Spatial Condition, Go Straight')
                selected_node=self.continue_straight(ob, t)            
        
        elif self.current_node['intersection_index'] == -1: # dead end
            # print('Spatial Reasoning On Dead-end Node')
            selected_node=self.continue_straight(ob, t)            
            self.dead_end+=1
      
        #check alternative stop 
        # if t>9 and self.stopping==False and len(self.stop_alternative_list)>0:
        if t>9 and self.stopping==False:
            alternative_node=self.check_alternative_stop(ob)
            if alternative_node is not None:
                selected_node=alternative_node
        #end check alternative stop

        print('[SpatialGPT Select Node:', selected_node['viewpointId'])
        # print('dead end num:', self.dead_end)
        # print('Total Backtrack Times:', self.num_backtrack)
        # print('Total Reflect Times', self.num_reflect)
        print('Latest Trajectory:')
        for node in self.Trajectory:
            print(node['index'])
        if selected_node is not None:
            self.current_viewIndex=int(selected_node['absolute_heading']/ math.radians(30))
            # print('its absolute heading:', selected_node['absolute_heading'])
            # print('its viewIndex:', self.current_viewIndex)
            self.SpatialKnowledgeGraph.add_edge(ob['viewpoint'], selected_node['viewpointId'])

        return selected_node
    
    def check_alternative_stop(self, ob):
        print('Entering Late Stage of Navigation, Checking Alternative Stop List:', self.stop_alternative_list)
        selected_alt_viewpoint=None
        selected_node=None
        prompt_system, prompt_user=None, None
        # if floor change detected, clear the stop alternative list
        if self.floor_changed==True:
            print('Floor Change Detected, Clear the Stop Alternative List')
            self.stop_alternative_list.clear()
        #     if len(self.stop_alternative_list)>0:
        #         print('Floor Change Detected, Stop Alternative List Before Keeping the Last One:', self.stop_alternative_list)
        #         self.stop_alternative_list=[self.stop_alternative_list[-1]]
        if len(self.stop_alternative_list)==1:             
            selected_alt_viewpoint=self.stop_alternative_list[0][0]
            print('Only One Alternative Viewpoint in the List, Selected Alternative Viewpoint:', selected_alt_viewpoint)
        else:    # >1 or ==0    
            if len(self.stop_alternative_list)>1:
                print('Multiple Stop Alternatives in the List:', self.stop_alternative_list)
                alt_traj_string=""
                #construct the alternative stop candidates string for prompt, which includes the display id of the candidate viewpoints in stop_alternative_list
                for alt_info in self.stop_alternative_list:
                    alt_viewpoint=alt_info[0] # get the viewpointId from the tuple (viewpointId, height, stop_distance)
                    # alt_height=alt_info[1] # get the height from the tuple (viewpointId, height, stop_distance)
                    # alt_stop_distance=alt_info[2] # get the stop_distance from the tuple (viewpointId, height, stop_distance)
                    # if the gap between floor_ref_z and alt_height is higher than 2 meters, ignore this candidate
                    # if abs(self.floor_ref_z - alt_height) > self.floor_delta_thresh:
                    #     print(f"Alternative viewpoint {alt_viewpoint} with height {alt_height:.2f}m is ignored due to large height gap from floor reference.")
                    #     continue
                    alt_display_id=None
                    node_location_path = os.path.join("bevbuilder", "bev", "node_location.txt")
                    if os.path.exists(node_location_path):
                        with open(node_location_path, 'r') as f:
                            node_locations = json.load(f)
                            all_nodes = node_locations.get("trajectory", [])
                            for node in all_nodes:  
                                if node.get('viewpointId') == alt_viewpoint:
                                    alt_display_id = node.get('display_number')
                                    break
                    if alt_display_id is not None:
                        alt_traj_string+=str(alt_display_id)+","
                if alt_traj_string!="":
                    alt_traj_string=alt_traj_string[:-1] # remove the last ,
            
                prompt_system, prompt_user=self.prompt_manager.make_stop_alternative_prompt(self.extracted_instruction, alt_traj_string)        
               
            
            elif len(self.stop_alternative_list)==0: # llm select from current floor
                print('No Alternative Viewpoint in the List')    
                prompt_system, prompt_user=self.prompt_manager.make_stop_all_alternative_prompt(self.extracted_instruction)        
            # print('stop alternative prompt:', prompt_system, prompt_user)        
            image_list = []
            image_list.clear()
            # append color and semantic bev map for stop alternative decision
            img_path = os.path.join("bevbuilder/bev/map_color.png")

            #verify the image path exists 
            if not os.path.exists(img_path):
                print(f"Error: BEV map image file not found at {img_path}")
                
            image_list.append(img_path)
            # append semantic bev map for stop alternative decision
            sem_img_path = os.path.join("bevbuilder/bev/map_semantic_legend.png")
            #verify the image path exists 
            if not os.path.exists(sem_img_path):
                print(f"Error: BEV map image file not found at {sem_img_path}")
            image_list.append(sem_img_path)

            if self.args.response_format == 'json':
                    # These full-floor BEVs can exceed the VLM context window. Resize
                    # temporary copies only for this late-stage alternative-stop call;
                    # all other inference paths continue to use the original images.
                    from PIL import Image

                    with tempfile.TemporaryDirectory(prefix="alternative_stop_bev_") as temp_dir:
                        resized_image_list = []
                        for image_index, image_path in enumerate(image_list):
                            with Image.open(image_path) as image:
                                image = image.convert("RGB")
                                image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                                resized_path = os.path.join(
                                    temp_dir, f"bev_{image_index}.png"
                                )
                                image.save(resized_path, format="PNG", optimize=True)
                            resized_image_list.append(resized_path)

                        alt_result, tokens = self.infer_strict_gpt_json(
                            prompt_system,
                            prompt_user,
                            resized_image_list,
                            "alternative_stop",
                            "check_alternative_stop",
                        )
                    print('[GPT Stop Alternative Inferenece by VLN-BEV]')
                    print('      Selected Stop Viewpoint:'+ alt_result['Destination_Viewpoint'])
                    print('      Reasoning:'+alt_result['Reasoning'])
                    selected_alt_display_id=alt_result['Destination_Viewpoint']
                    if selected_alt_display_id!="none":
                        node_location_path = os.path.join("bevbuilder", "bev", "node_location.txt")
                        if os.path.exists(node_location_path):  
                            #get its viewpointId based on the display_number
                            with open(node_location_path, 'r') as f:
                                node_locations = json.load(f)
                                all_nodes = node_locations.get("trajectory", [])
                                for node in all_nodes:  
                                    if node.get('display_number') == int(selected_alt_display_id):
                                        selected_alt_viewpoint = node.get('viewpointId')
                                        break
                        else:
                            print(f"Error: Node location file not found at {node_location_path}")
        # plan the shortest path to the selected alternative stop viewpoint
        if selected_alt_viewpoint is not None:
            print('Selected Alternative Viewpoint from Stop Alternative List:', selected_alt_viewpoint)
            try:
                self.path_to_alt = nx.shortest_path(
                    self.SpatialKnowledgeGraph,
                    source=self.current_node['index'],
                    target=selected_alt_viewpoint,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.path_to_alt = None
            if self.path_to_alt is not None and len(self.path_to_alt) > 1:
                next_viewpoint = self.path_to_alt[1]  # the next step towards the alternative viewpoint
                # remove the first node in the path which is the current node
                self.path_to_alt = self.path_to_alt[1:]
                for candidate in ob['candidate']:
                    if candidate['viewpointId'] == next_viewpoint:
                        selected_node = candidate
                        self.back_to_stop=True
                        print('Start Backtracking to Alternative Stop - Next Node:', selected_node['viewpointId'])
                        break                    
            else:
                print('No valid path to the selected alternative viewpoint or already at the alternative viewpoint.')
        
        return selected_node

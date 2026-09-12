import re
import math
import os
from tkinter.filedialog import test

class OneStagePromptManager(object):
    def __init__(self, args):

        self.args = args
        self.history  = ['' for _ in range(self.args.batch_size)]
        self.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.graph  = [{} for _ in range(self.args.batch_size)]
        self.trajectory = [[] for _ in range(self.args.batch_size)]
        self.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

    def get_action_concept(self, rel_heading, rel_elevation):
        if rel_elevation > 0:
            action_text = 'go up'
        elif rel_elevation < 0:
            action_text = 'go down'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    action_text = 'turn left'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn right'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    action_text = 'turn right'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn left'
            elif rel_heading == 0:
                action_text = 'go forward'

        return action_text

    def make_action_prompt(self, obs, previous_angle):

        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs

        batch_view_lens, batch_cand_vpids = [], []
        batch_cand_index = []
        batch_action_prompts = []

        for i, ob in enumerate(obs):
            cand_vpids = []
            cand_index = []
            action_prompts = []

            if ob['viewpoint'] not in nodes_list[i]:
                # update nodes list (place 0)
                nodes_list[i].append(ob['viewpoint'])
                node_imgs[i].append(None)

            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            # cand views
            for j, cc in enumerate(ob['candidate']):

                cand_vpids.append(cc['viewpointId'])
                cand_index.append(cc['pointId'])
                direction = self.get_action_concept(cc['absolute_heading'] - previous_angle[i]['heading'],
                                                          cc['absolute_elevation'] - 0)

                if cc['viewpointId'] not in nodes_list[i]:
                    nodes_list[i].append(cc['viewpointId'])
                    node_imgs[i].append(cc['image'])
                    node_index = nodes_list[i].index(cc['viewpointId'])
                else:
                    node_index = nodes_list[i].index(cc['viewpointId'])
                    node_imgs[i][node_index] = cc['image']

                action_text = direction + f" to Place {node_index} which is corresponding to Image {node_index}"
                action_prompts.append(action_text)

            batch_cand_index.append(cand_index)
            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids

        return {
            'cand_vpids': batch_cand_vpids,
            'cand_index':batch_cand_index,
            'action_prompts': batch_action_prompts,
        }

    def make_action_options(self, cand_inputs, t):
        action_options_batch = []  # complete action options
        only_options_batch = []  # only option labels
        batch_action_prompts = cand_inputs["action_prompts"]
        batch_size = len(batch_action_prompts)

        for i in range(batch_size):
            action_prompts = batch_action_prompts[i]
            if bool(self.args.stop_after):
                if t >= self.args.stop_after:
                    action_prompts = ['stop'] + action_prompts

            full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
            only_options = [chr(j + 65) for j in range(len(action_prompts))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""

    def make_map_prompt(self, i):
        # graph-related text
        trajectory = self.trajectory[i]
        nodes_list = self.nodes_list[i]
        graph = self.graph[i]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f""" {node_index}"""

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f""" {adj_index},"""

                graph_text += f"""\nPlace {node_index} is connected with Places{adj_text}"""[:-1]

        # ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):

            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"""\nPlace {node_index}, which is corresponding to Image {node_index}"""

        if supp_exist is None:
            graph_supp_text = """Nothing yet."""

        return trajectory_text, graph_text, graph_supp_text
    
    def make_bev_align_prompt(self, instru, traj_string):     
#         prompt_system = f"""
# You are a spatial reasoning expert assisting a Vision-Language Navigation (VLN) agent. You are provided below a navigation instruction, a BEV (Bird's-Eye View) map showing top-down view of the environment. Your task is to determine the best next viewpoint for the agent to move to.

# Explanation of Inputs:
# 1. Navigation Instruction: describes step-by-step guidance for navigation
# 2. BEV Map:
#    - Blue lines: traversed trajectory.
#    - Blue nodes (numbered): visited viewpoints in order ({traj_string}).
#    - Red node: current agent position.
#    - Green nodes (numbered): unvisited viewpoints.
#    - Green dotted lines: unvisited paths to unvisited viewpoints.

# Reasoning Process:
# Select ONE candidate viewpoint connected to the red node, based on the strategy below：

# If EARLY STAGE (i.e., the trajectory is short in the BEV), you should prefer invisible area exploration using indoor layout commonsense knowledge.
# Follow priority:
# 1. If a staircase is visible and reachable → MUST select it
# 2. Otherwise, choose a viewpoint leading to room entrances, corridors, or turns (new regions)
# 3. Otherwise, choose the viewpoint that moves towards reasonable area  (e.g., bedroom lead to bathroom, living room near to kitchen)

# If LATE STAGE (i.e., the trajectory is long in the BEV), you should focus on looking for the target room and object matching the instruction.
# - If a target room is identified:
#   - enter it
#   - move closer to the target object
# - If a room appears incorrect:
#   - exit quickly
#   - move toward a new room or branch

# Output Format(JSON):
# {{
#   "Next_Viewpoint": "<node id>",
#   "Reasoning": "<concise reasoning explanation combining instruction intent and local spatial evidence>"
# }}
# """
        prompt_system = f"""
You are a spatial reasoning expert assisting a Vision-Language Navigation (VLN) agent. You are provided below a navigation instruction, a BEV (Bird's-Eye View) map showing top-down view of the environment. Your task is to determine the best next viewpoint for the agent to move to.

Explanation of Inputs:
1. Navigation Instruction: describes step-by-step guidance for navigation
2. BEV Map:
   - Blue lines: traversed trajectory.
   - Blue nodes (numbered): visited viewpoints in order ({traj_string}).
   - Red node: current agent position.
   - Green nodes (numbered): unvisited candidate viewpoints.
   - Green dotted lines: paths to unvisited candidate viewpoints.

Reasoning Process:
Select ONE candidate viewpoint directly connected to the red node according to the following priority:
1. Choose a candidate viewpoint leading to corridors or turns that open access to new regions.
2. If the agent is at a doorway and the visible room does not appear to be the target room, immediately leave and explore a different room or branch. If the visible room appears to be the target room, enter it and carefully search for the target object.
3. You can also use indoor layout commonsense based on the given instruction.  (e.g., a bedroom may lead to a bathroom, and a living room is often near a kitchen).

Output Format(JSON):
{{
  "Next_Viewpoint": "<node id>",
  "Reasoning": "<concise reasoning explanation combining instruction intent and local spatial evidence>"
}}
"""
        prompt_user = f""" Navigation Instruction:{instru}."""
        return prompt_system, prompt_user
    
    def make_frontier_prompt(self, step_instru, current_node, SKG):
        # prompt_system = f"""You are an indoor navigation assistant. The agent is currently navigating a structured environment based on a natural language instruction.A Spatial Knowledge Graph (SKG) has been constructed from past observations,containing all visited and unvisited nodes, along with semantic landmarks and connectivity. The agent must determine the next node to explore based on the instruction and current SKG state. Your task is to infer which unvisited node as exploration frontier the agent should move to next. You can use your commonsense knowledge about indoor environments for this reasoning. Provide the output in JSON format: ’Selected Frontier’:”, ’Reasoning’:’ ’."""
        prompt_system = f"""You are an indoor navigation assistant. The agent is currently navigating a structured environment based on a natural language instruction. A Spatial Knowledge Graph (SKG) has been constructed from past observations,containing all visited and unvisited nodes, along with semantic landmarks and connectivity. Your are provided (1) a navigation instruction (2) current SKG (3) A bird’s-eye view (BEV) image  showing a top-down color view of the environment. BEV also includes the agent’s trajectory, where solid blue lines are the traversed paths; blue nodes mark visited viewpoints with numeric labels; the red node marks the agent's current viewpoint.  The agent must determine the next node to backtrack and explore based on the navigation instruction, BEV and current SKG state. Your task is to infer which unvisited node as exploration frontier the agent should move to next. You can use your commonsense knowledge about indoor environments for this reasoning. Provide the output in JSON format: ’Selected_Frontier’:”, ’Reasoning’:’ ’."""
        prompt_user = f"""Instruction:{step_instru}. Current Node:{current_node}. SKG:{SKG} """
        return prompt_system, prompt_user
    
    def make_stop_all_alternative_prompt(self, step_instru):
        
        prompt_system = f"""Background: 
You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent in an indoor environment. The agent is in the final stage of navigation, and the BEV for the current floor has already been formed. The agent has missed the destination and continued exploring.
You are given:
(1) A navigation instruction describing the route and the destination
(2) A color BEV image (top-down view with spatial layout and trajectory)
(3) A semantic BEV image (top-down view with semantic labels)
Both BEVs correspond to the current floor and show the agent's trajectory on this floor:
• Solid blue lines indicate traversed paths
• Blue nodes indicate visited viewpoints with node IDs
• The red node indicates the agent's current viewpoint

Task:
Select one blue node, with node ID greater than 2, that is the most likely missed destination viewpoint according to the navigation instruction.

Reasoning Guidelines:
• Since the two BEV images represent the same scene, use them jointly and consistently. Use the semantic BEV mainly to identify objects, landmarks, rooms, or functional areas mentioned in the instruction. Use the color BEV mainly to understand spatial layout, geometry, and trajectory context.
• Treat the trajectory only as auxiliary exploration context, not as strict evidence of instruction progress.
• Reason in two steps:
    1. infer the most likely destination location in the BEV which is spatially plausible and semantically consistent with the destination description. 
    2. identify the blue node whose position best matches the inferred destination location and description.

Output Format(JSON):
{{
  "Destination_Viewpoint": "<node id>",
  "Reasoning": "<concise explanation based on instruction and BEV evidence>"
}}
"""
        prompt_user = f""" Navigation Instruction:{step_instru}."""       

        return prompt_system, prompt_user
    

    def make_stop_alternative_prompt(self, step_instru, stop_alternatives):
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent in an indoor environment. The agent has followed the instruction and passed multiple potential destination locations along its trajectory.

You are given:
(1) A navigation instruction describing the route and the destination
(2) A color BEV image (top-down view with spatial layout and trajectory)
(3) A semantic BEV image (top-down view with semantic labels)
(4) A list of candidate viewpoint nodes (node IDs separated by commas)
Both BEVs correspond to the current floor and show the agent's trajectory on this floor:
• Solid blue lines indicate traversed paths
• Blue nodes indicate visited viewpoints with node IDs
• The red node indicates the agent’s current viewpoint

Task:
Select one node from the candidate list that best matches the destination described in the navigation instruction.

Reasoning Guidelines
• Since the two BEV images represent the same scene, use them jointly and consistently. Use the semantic BEV mainly to identify objects, landmarks, rooms, or functional areas mentioned in the instruction. Use the color BEV mainly to understand spatial layout, geometry, and trajectory context.
• Align the instruction with the trajectory: 1) Infer where the destination should be relative to the path. 2) Identify which candidate node best matches the described destination
• Choose the node that best satisfies both semantic consistency and spatial alignment.

Output Format(JSON):
{{
  "Destination_Viewpoint": "<node id>",
  "Reasoning": "<concise explanation based on instruction and BEV evidence>"
}}
"""
        prompt_user = f""" Navigation Instruction:{step_instru}. Candidate Viewpoints:{stop_alternatives}."""

        return prompt_system, prompt_user
    
    def make_stop_bev_approach_prompt(self, step_instru, traj_string):

        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning in an indoor environment.

You are given:
(1) A navigation instruction, offering navigation stop guidance with the destination description
(2) A color BEV (Bird's-Eye View) image showing a top-down color view of the environment.
(3) A semantic BEV image showing the same top-down view with semantic labels.

Both BEV images include the agent's trajectory:
• Solid blue lines: traversed paths
• Blue nodes: visited viewpoints with numeric labels. Their chronological order is: {traj_string}
• Red node: the agent's current viewpoint, which is at the center of a red dashed circle with 2.5 meters radius.


Your task is to decide whether the agent should stop NOW at the CURRENT viewpoint.

Reasoning guidance:
• Since the two BEV images represent the same scene, use them jointly and consistently. Use the semantic BEV mainly to identify objects, landmarks, rooms, or functional areas mentioned in the instruction. Use the color BEV mainly to understand spatial layout, geometry, and trajectory context.
• Based on the destination description in the navigation instruction, locate the destination position in the BEV. If the destination cannot be found in the BEV, set "stop_now" = "no".
• If the located destination position lies inside the red dashed circle, set "stop_now" = "yes". Otherwise, set "stop_now" = "no".

Output format (strict JSON):
{{
  "stop_now": "",
  "confidence": "",
  "brief_reason": ""
}}

Output requirements  (strict JSON):
1. "stop_now" must be "yes" or "no".
2. "confidence" must be a float between 0.0 and 1.0.
3. "brief_reason" should describe the destination position and explain why the agent should stop or continue based on the reasoning guidance.
"""
        prompt_user = f""" Navigation Instruction:{step_instru}."""
        return prompt_system, prompt_user


    
    def make_stop_bybev2_prompt(self, step_instru, traj_string):

      
        # add semantic bev using map_stop_semantic_legend for stop
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning in an indoor environment. You are positioned at a viewpoint near the destination. You are provided with the following information: (1) A navigation instruction, offering navigation stop guidance. (2) A color bird’s-eye view (BEV) image showing a top-down color view of the environment. (3) A semantic BEV image showing a top-down semantic map with semantic labels for better environment understanding. The color BEV and semantic BEV are two different representations of the same scene and the same agent state.
    Both BEV images include the agent’s trajectory:
    • Solid blue lines indicate traversed paths.
    • Blue nodes mark visited viewpoints with numeric labels.
    • Their chronological order is: {traj_string}.
    • The red node marks the agent’s current viewpoint, which is at the center of a circle with 3 meters radius.
    Your task is to judge:
    1. Whether the destination (or landmark) described in the navigation instruction lies inside that circle.
    2. Whether the agent should stop now.
    Reasoning guidance
    • Since the two BEV images represent the same scene, use them jointly and consistently.
    • Use the semantic BEV mainly to identify objects, landmarks, rooms, or functional areas mentioned in the instruction.
    • Use the color BEV mainly to understand spatial layout, geometry, and trajectory context.
    Output format (strict JSON):
    {{
    "destination_inside_circle": "",
    "stop_now": "",
    "confidence": "",
    "brief_reason": ""
    }}
    Note:
    1. "destination_inside_circle" must be "yes" or "no", representing the agent is within 3 meters of the destination.
    2. "stop_now" must be "yes" or "no".
    3. "confidence" must be a float between 0.0 and 1.0 indicating your certainty.
    4. "brief_reason" should concisely explain the judgment based on the instruction and the two BEV representations.
    """
        #blurred circle prompt using map_stop_color for stop
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning. You are positioned at a viewpoint near destination within an indoor environment.You are provided below with (1) a navigation instruction, offering navigation stop guidance (2) A bird’s-eye view (BEV) image that also includes the agent’s trajectory, where solid blue lines are the traversed paths; blue nodes mark visited viewpoints with numeric labels; their chronological order is {traj_string}; the red node marks the agent's current viewpoint; the red node is at the center of a circle with 3 meters radius.  Your task is to judge whether the destination (or landmark) described in the navigation instruction lies inside that circle, and whether the agent should stop now. Your response should be in JSON format as follows: "destination_inside_circle":" ", "stop_now":" ", "confidence":" ","brief_reason":" ".      Note: (1) destination_inside_circle and stop_now must be filled with "yes" or "no" based on your judgement. (2) confidence must be a float between 0.0 and 1.0 expressing the model's certainty. (3) brief_reason should explain your judgment."""

        prompt_user = f""" Navigation Instruction:{step_instru}."""
        return prompt_system, prompt_user
    
    def make_stop_bybev_prompt(self, step_instru, traj_string):

        #blurred circle prompt using map_stop_color for stop
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning. You are positioned at a viewpoint near destination within an indoor environment.You are provided below with (1) a navigation instruction, offering navigation stop guidance (2) A bird’s-eye view (BEV) image that also includes the agent’s trajectory, where solid blue lines are the traversed paths; blue nodes mark visited viewpoints with numeric labels; their chronological order is {traj_string}; the red node marks the agent's current viewpoint; the red node is at the center of a circle with 3 meters radius.  Your task is to judge whether the destination (or landmark) described in the navigation instruction lies inside that circle, and whether the agent should stop now. Your response should be in JSON format as follows: "destination_inside_circle":" ", "stop_now":" ", "confidence":" ","brief_reason":" ".      Note: (1) destination_inside_circle and stop_now must be filled with "yes" or "no" based on your judgement. (2) confidence must be a float between 0.0 and 1.0 expressing the model's certainty. (3) brief_reason should explain your judgment."""

        prompt_user = f""" Navigation Instruction:{step_instru}."""
        return prompt_system, prompt_user
    
    def make_synchronize_bybev_prompt(self, step_instru, prev_instru, traj_string):
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning. 
        # You are provided below with (1) a sequence of stepwise navigation instructions. (2) the previous step instruction that the agent was performing, indicating the movement from the previous observation point to the current observation point. (3) a BEV (bird's-eye view) image with a perception radius of 7 meters.    
        # Your task is to judge whether the agent has completed the previous step instruction (i.e. has it already {prev_instru}) and provide a brief reason. 
        # Your response should be in JSON format as follows: 
        #     "completed_status":" ", 
        #     "confidence":" ", 
        #     "brief_reason":" ". 
        # Note: 
        # (1) completed_status must be filled with "yes" or "no" based on whether the previous step is judged completed. 
        # (2) confidence must be a float between 0.0 and 1.0 expressing the model's certainty. 
        # (3) Do NOT invent landmarks, distances, or layouts not visible in the provided BEV image or the instruction text. 
        # (4) brief_reason should provide a single concise sentence that summarizes the BEV image and justifies the completed_status judgment.
        # """

        
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning. 
        # You are provided below with (1) a sequence of stepwise navigation instructions. (2) the previous step instruction that the agent was performing, indicating the movement from the previous observation point to the current observation point. (3) A bird’s-eye view (BEV) image that also includes the agent’s trajectory, where solid blue lines are the traversed paths; blue filled circles mark each visited viewpoint with numeric labels; their chronological order is {traj_string}; the red filled circle marks the agent's current viewpoint. 
        # Your task is to judge whether the agent has completed the previous step instruction (i.e. has it already {prev_instru}) and provide a brief reason. 
        # Your response should be in JSON format as follows: 
        #     "completed_status":" ", 
        #     "confidence":" ", 
        #     "brief_reason":" ". 
        # Note: 
        # (1) completed_status must be filled with "yes" or "no" based on whether the previous step is judged completed. 
        # (2) confidence must be a float between 0.0 and 1.0 expressing the model's certainty. 
        # (3) Do NOT invent landmarks, distances, or layouts not visible in the provided BEV image or the instruction text.
        # (4) If the previous step instruction involves only a turning action without any related landmark, there is no need for judgment, simply fill the completed_status field with "yes".  
        # (5) brief_reason should summarize the BEV image and explain your judgment.
        # """

        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with multimodal spatial reasoning. 
        You are provided below with (1) a sequence of stepwise navigation instructions. (2) the previous step instruction that the agent was performing, indicating the movement from the previous observation point to the current observation point. (3) A bird’s-eye view (BEV) image that also includes the agent’s trajectory, where solid blue lines are the traversed paths; blue filled circles mark each visited viewpoint with numeric labels; their chronological order is {traj_string}; the red filled circle marks the agent's current viewpoint. 
        Your task is to judge whether the agent has completed the previous step instruction (i.e. has it already {prev_instru}) and provide a brief reason. 
        Your response should be in JSON format as follows: 
            "completed_status":" ", 
            "confidence":" ", 
            "brief_reason":" ". 
        Note: 
        (1) completed_status must be filled with "yes" or "no" based on whether the previous step is judged completed. 
        (2) confidence must be a float between 0.0 and 1.0 expressing the model's certainty. 
        (3) Use the trajectory as supporting context to determine whether the agent has already achieved the spatial relation with the landmark mentioned in the previous instruction step.
        (4) If the previous step instruction involves only a turning action without any related landmark, there is no need for judgment, simply fill the completed_status field with "yes".  
        (5) brief_reason should summarize the BEV image and explain your judgment.
        """
        prompt_user = f""" Stepwise Navigation Instruction: {step_instru}. Previous Step Instruction: {prev_instru}"""
        return prompt_system, prompt_user
    

    def make_synchronize_bydcel_prompt(self, step_instru, prev_instru, dcel):
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are provided below with (1) a sequence of stepwise navigation instructions. (2) the previous step instruction that the agent was performing, indicating the movement from the previous observation point to the current observation point.. (3) The landmarks viewed by the agent, grouped according to their relative direction to the agent.    Your task is to judge whether the agent has completed the previous step instruction (i.e. has it already {prev_instru})?  and provide a brief reason. Your response should be in JSON format as follows:    "completed_status": " ", "brief_reason":" " . Note: (1) completed_status field should be filled with "yes" or "no" based on your judgment on the previous step instruction's completion status. (2) The viewed landmarks' relative direction to the agent is categorized into four areas: front, back, left, and right. One area may contains multiple landmarks separated by comma. (3) If the previous step instruction involves only a turning action without any related landmark, there is no need for judgment, simply fill the completed_status field with "yes". (4) If the previous step instruction's related landmark is mentioned again in the subsequent step instruction, there is no need for judgement, simply fill the completed_status field with "yes". (5) For normal indoor object landmarks in the previous step, such as tables or doors, their appearance in the agent's front area indicates that the agent has not yet completed the previous step of "passing" or "going through" the landmark. Conversely, their appearance in the agent's back area indicates that the agent has already completed the previous step of "passing" or "going through" the landmark. For large-field landmarks from the previous step, where the agent can be inside (e.g., a room or path), if the landmark appears in multiple areas around the agent, it indicates that the agent has completed the previous step of "entering" the large-field landmark but has not yet completed the step of "exiting" the large-field landmark. If such a large-field landmark from the previous step appears only in the back area of the agent, it signifies that the agent has completed the previous step of "exiting" this large-field landmark. """
        prompt_user = f""" Stepwise Navigation Instruction:{step_instru}. Previous Step Instruction:{prev_instru}. Landmarks' Relative Direction:{dcel}"""
        return prompt_system, prompt_user

    def make_gpt_stop_distance_reason_prompt(self, ob, instruction, navigable_images):
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center coordinates of the destination object (in pixels, relative to the image top-left corner).  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see. The image's full dimensions: width = 640 pixels, height = 480 pixels. You should carefully identify the destination object's position in this image, then fill the dest_x and dest_y fields with the center coordinates of the destination object, measured relative to the top-left corner of the image.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination object, estimate the distance in meters between you and the destination object, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        prompt_user = f""" Navigation Instruction:{instruction}"""
        return prompt_system, prompt_user

    def make_gpt_stop_topology_reason_prompt(self, ob, instruction, navigable_images):
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center coordinates of the destination object (in pixels, relative to the image top-left corner).  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see. The image's full dimensions: width = 640 pixels, height = 480 pixels. You should carefully identify the destination object's position in this image, then fill the dest_x and dest_y fields with the center coordinates of the destination object, measured relative to the top-left corner of the image.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) Based on the provided environment images, if you believe the agent has already reached the destination described by the instruction (i.e. at the correct topological stop location), fill the stop_distance field with "0.8". Otherwise, fill the stop_distance field with "-1".  """
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) Based on the provided 12 directional images, if you believe the agent has already reached the destination described by the instruction (i.e. aligned with the correct topological destination viewpoint), fill the stop_distance field with "0.8". Otherwise, fill the stop_distance field with "-1".  """
        prompt_user = f""" Navigation Instruction:{instruction}"""
        return prompt_system, prompt_user
    
    def make_gpt_breadth_distance_reason_prompt(self, ob, instruction, orientation, navigable_images, progress):
        left_1 = (orientation - 2) % 12
        left_2 = (orientation - 3) % 12
        right_1 = (orientation + 2) % 12
        right_2 = (orientation + 3) % 12
        forward_1 = (orientation + 1) % 12
        forward_2 = (orientation - 1) % 12
        back_1 = (orientation + 5) % 12
        back_2 = (orientation + 6) % 12
        back_3 = (orientation + 7) % 12
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center coordinates of the destination object (in pixels, relative to the image top-left corner).  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center pixel coordinates of the destination object, measured relative to the top-left corner of the image (image dimensions: width = 640 pixels, height = 480 pixels).  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see.The image's full dimensions: width = 640 pixels, height = 480 pixels. You should carefully identify the destination object's position in this image, then fill the dest_x and dest_y fields with the center coordinates of the destination object, measured relative to the top-left corner of the image.   If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination object described in the navigation instruction in any image, estimate the distance in meters between you and the destination object, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a short path plan and the detailed reason why the selected image got the highest similarity score as next point to move towards. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        prompt_user = f""" Navigation Instruction:{instruction} Estimated Progress:{progress}."""
        return prompt_system, prompt_user

    def make_gpt_breadth_topology_reason_prompt(self, ob, instruction, orientation, navigable_images, progress):
        left_1 = (orientation - 2) % 12
        left_2 = (orientation - 3) % 12
        right_1 = (orientation + 2) % 12
        right_2 = (orientation + 3) % 12
        forward_1 = (orientation + 1) % 12
        forward_2 = (orientation - 1) % 12
        back_1 = (orientation + 5) % 12
        back_2 = (orientation + 6) % 12
        back_3 = (orientation + 7) % 12
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center coordinates of the destination object (in pixels, relative to the image top-left corner).  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see, fill the dest_x and dest_y fields with the center pixel coordinates of the destination object, measured relative to the top-left corner of the image (image dimensions: width = 640 pixels, height = 480 pixels).  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'dest_image':' ', 'dest_obj':' ', 'dest_x':' ', 'dest_y':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number, fill the dest_image field with the image ID in which you see the destination, fill the dest_obj field with the destination object you see.The image's full dimensions: width = 640 pixels, height = 480 pixels. You should carefully identify the destination object's position in this image, then fill the dest_x and dest_y fields with the center coordinates of the destination object, measured relative to the top-left corner of the image.   If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination object described in the navigation instruction in any image, estimate the distance in meters between you and the destination object, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        # prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) Based on the provided environment images, if you believe the agent has already reached the destination described by the instruction (i.e. at the correct topological stop location), fill the stop_distance field with "0.8". Otherwise, fill the stop_distance field with "-1".  """ 
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) Based on the provided 12 directional images, if you believe the agent has already reached the destination described by the instruction (i.e. aligned with the correct topological destination viewpoint), fill the stop_distance field with "0.8". Otherwise, fill the stop_distance field with "-1".  """ 
        prompt_user = f""" Navigation Instruction:{instruction} Estimated Progress:{progress}."""
        return prompt_system, prompt_user

    
#     def make_spatial_extract_reverie_instruction_prompt(self, obs):
#         prompt_system = f"""
# You are a spatial reasoning expert for a Vision-Language Navigation (VLN) agent. You are given one navigation instruction. Your job is to convert it into a structured spatial plan.
# You must complete THREE tasks.
# ================================
# Task 1: Infer a preparatory subgoal
# ================================
# Infer a reasonable preparatory subgoal that the agent should reach BEFORE the first landmark explicitly mentioned in the given instruction.

# Requirements for the subgoal:
# 1. The subgoal must NOT be the same as the first landmark in the instruction.
# 2. The subgoal should be inferred using commonsense indoor spatial knowledge, e.g. go to bedroom before reaching the bathroom, go through staircase to the next floor.
# 3. The subgoal should help the agent spatially approach the first landmark in a natural way.
# 4. If no reasonable inferred subgoal is possible, use "corridor or staircase" as the default subgoal.

# Then add ONE short sentence at the beginning of the original instruction to guide the agent to first move toward this inferred subgoal.

# The added sentence must:
# - be concise
# - be spatially meaningful
# - clearly indicate that the agent should first go toward the inferred subgoal

# Fill the "full_instruction" with the FULL modified instruction, including the added subgoal sentence.
# ================================
# Task 2: Decompose the full instruction into stepwise actions
# ================================
# Split the FULL modified instruction, including the added subgoal sentence, into a time-ordered sequence of stepwise actions.

# Rules:
# 1. Each action should correspond to one atomic movement, navigation operation, or spatially meaningful verb phrase.
# 2. Preserve the original wording.
# 3. Keep the temporal order exactly.
# 4. For each action, use the exact action phrase from the modified instruction as the "action_name".

# For each action, extract referenced landmarks.

# Each action must contain:
# - "action_name": exact phrase from the modified instruction
# - "landmarks": a list of landmark objects

# Each landmark object must contain:
# - "landmark_name"
# - "relative_spatial_area": one of ["front", "back", "left", "right"]

# Rules for landmarks:
# 1. If an action refers to no landmark, use an empty list [] for the "landmarks".
# 2. An action may refer to multiple landmarks.
# 3. Map each landmark relation into one of the four categories only: front, back, left, right.

# ================================
# Task 3: Detect elevation-decreasing actions
# ================================
# Determine whether the navigation involves elevation-decreasing movement (e.g., go downstairs, step down).
# If yes:
# - Return a list of step indices (0-based) corresponding to those actions. Step indices must match the order in "step_actions" of the Task 2's result.
# If none:
# - Return an empty list [].

# ================================
# Output Format
# ================================
# Output STRICTLY as valid JSON only.
# {{
#   "full_instruction": "...",
#   "elevation_decrease": [ ... ],
#   "step_actions": [
#     {{
#       "action_name": "...",
#       "landmarks": [
#         {{
#           "landmark_name": "...",
#           "relative_spatial_area": "front | back | left | right"
#         }}
#       ]
#     }}
#   ]
# }}
# """
#         prompt_user = f"""Instruction-{obs[0]["instruction"]}"""

#         return prompt_system, prompt_user
    
    def make_spatial_extract_instruction_prompt(self, obs):
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have two task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The results should be output as JSON in the following struture: 'elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (2) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (3) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (4) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (5) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have three task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The third task is to classify the stop condition in the final segment of the instruction as either an object-relative stop or a topology-based stop. For Object-Relative Stop: The agent should stop based on proximity to a clearly identifiable object. This type can be resolved using object detection and depth estimation. Examples: "Stop next to the sink", "Wait by the fireplace", "Stop in front of the bed". For Topology-Based Stop, the agent should stop at a location defined by spatial layout or room connectivity (e.g., doorways, halls, corners, or areas relative to rooms). Object detection is insufficient; structural reasoning is required. Examples: "Stop in the hallway", "Stop at the end of the corridor", "Stop in the room left of the bathroom".   The results should be output as JSON in the following struture: 'stop_type':' ','elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1) The stop_type field should be filled with "0" for an object-relative stop or "1" for a topology-based stop.  (2)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (3) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (4) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (5) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (6) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have three task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The third task is to classify the stop condition in the final segment of the instruction as either an object-relative stop or a topology-based stop. For Object-Relative Stop: The agent should stop based on proximity to a clearly identifiable object(e.g., couch, table, stair). This type can be resolved using object detection and depth estimation. Examples: "Stop next to the sink", "Wait by the fireplace", "Stop on the sixth step from the bottom". For Topology-Based Stop, the agent should stop at a location defined by spatial layout or room connectivity (e.g., doorways, halls, corners, or areas relative to rooms). Object detection is insufficient; structural reasoning is required. Examples: "Stop in the hallway", "Stop at the end of the corridor", "Stop in the room left of the bathroom".   The results should be output as JSON in the following struture: 'stop_type':' ','elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1) The stop_type field should be filled with "0" for an object-relative stop or "1" for a topology-based stop.  (2)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (3) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (4) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (5) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (6) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have three task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The third task is to classify the stop condition in the final segment of the instruction as either an object-relative stop or a topology-based stop. For Object-Relative Stop: The agent should stop based on proximity to a clearly identifiable object. This type can be resolved using object detection and depth estimation. Examples: "Stop next to the sink", "Wait by the fireplace", "Stop in front of the bed". For Topology-Based Stop, the agent should stop at a location defined by spatial layout or room connectivity (e.g., doorways, stairs, corners, or areas relative to rooms), where object detection alone is insufficient and structural reasoning over the environment is required. Examples: "Stop in the hallway", "Stop at the end of the corridor", "Stop in the room left of the bathroom".   The results should be output as JSON in the following struture: 'stop_type':' ','elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1) The stop_type field should be filled with "0" for an object-relative stop or "1" for a topology-based stop.  (2)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (3) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (4) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (5) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (6) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        # prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have three task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The third task is to judge whether the stopping condition described at the end of the navigation instruction occurs on a stair step within a staircase, not at a stair end (i.e., the top or bottom landing). Examples: "Go up the stairs and stop on the sixth step from the bottom", "Go up three steps then stop","Stop on the fourth step up".  The results should be output as JSON in the following structure: 'stop_type':' ','elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1) The stop_type field should be filled with "1" if the stop occurs on a stair step within a staircase, and "0" otherwise.  (2)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (3) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (4) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (5) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (6) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction."""
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have three task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement, such as “go downstairs”, “go to the bottom of the stairs". If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The third task is to judge whether the stopping condition described at the end of the navigation instruction occurs on a stair step within a staircase, not at a stair end (i.e., the top or bottom landing). Examples: "Go up the stairs and stop on the sixth step from the bottom", "Go up three steps then stop","Stop on the fourth step up".  The results should be output as JSON in the following structure: 'stop_type':' ','elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1) The stop_type field should be filled with "1" if the stop occurs on a stair step within a staircase, and "0" otherwise.  (2)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (3) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (4) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (5) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (6) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        prompt_user = f"""Instruction-{obs[0]["instruction"]}"""

        return prompt_system, prompt_user
    
    def make_spatial_search_landmarks_prompt(self, orientation, step_instru, landmarks):
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a final step of navigation instruction to guide the agent to stop close to the destination. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If the detination is not visible in any image, fill the stop_distance field with "-1".  Your distance estimation is very important for stop decision, must make it very carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a full navigation instruction to guide the agent to the final destination and stop. You have two tasks. The first task is to compare the surrounding environment with the given instruction to assess execution progress. If the agent reaches the final step of the given instruction, determine whether it has arrived at the final destination based on visual inputs. The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you determine that you have reached the final step of the given instruction and you have arrived at the final destination, estimate the distance in meters between you and the final destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you determine that you have not yet reached the final step of the instruction or the final destination, fill the stop_distance field with "-1".  Your judgement is crucial for the stop decision, so make it carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a final step of navigation instruction to guide the agent to stop close to the destination. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If the detination is not visible in any image, fill the stop_distance field with "-1".  Your distance estimation is very important for stop decision, must make it very carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a final step of navigation instruction to guide the agent to stop close to the destination. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item must includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If the detination is not visible in any image, fill the stop_distance field with "-1".  Your distance estimation is very important for stop decision, must make it very carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) The 'presence' field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.  Your task is to search the specified landmarks provided below within these 12 images. Your response should be in JSON format with EXACTLY this structure: 
{{
  "results": [
   {{"image_id": 0, "result_list": []}},
   {{"image_id": 1, "result_list": []}},
   {{"image_id": 2, "result_list": []}},
   {{"image_id": 3, "result_list": []}},
   {{"image_id": 4, "result_list": []}},
   {{"image_id": 5, "result_list": []}},
   {{"image_id": 6, "result_list": []}},
   {{"image_id": 7, "result_list": []}},
   {{"image_id": 8, "result_list": []}},
   {{"image_id": 9, "result_list": []}},
   {{"image_id": 10, "result_list": []}},
   {{"image_id": 11, "result_list": []}} 
   ]
}}, where result_list is a nested key-value list in which each item must include only the field "landmark_name". Keep the wrapper key exactly as "results". Do NOT use other keys.  Note: 1) The JSON list should include exactly 12 items (image_id 0–11) for each image query result, even if no landmarks are detected. Each item must have "image_id" and "result_list". 2) The "result_list" should include only the landmarks that are actually detected in the image. The 'result_list' should be set as an empty list if no landmarks are detected in the image. 3) Do not include landmark_id, type, presence, explanations, or duplicate landmark names. """
        
        landmarks_str = "["
        for i in range(len(landmarks)):
            name = landmarks[i]
            landmarks_str += f"""{i+1}:{name};"""
        # prompt_user = f"""Final Step of Navigation Instruction:{step_instru}. Landmarks(ID:Name)-{landmarks_str}]"""
        prompt_user = f"""Landmarks(ID:Name)-{landmarks_str}]"""

        return prompt_system, prompt_user
  
    

    def parse_planning(self, nav_output):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        keyword1 = '\nNew Planning:'
        keyword2 = '\nAction:'
        for i in range(batch_size):
            output = nav_output[i].strip()
            start_index = output.find(keyword1) + len(keyword1)
            end_index = output.find(keyword2)

            if output.find(keyword1) < 0 or start_index < 0 or end_index < 0 or start_index >= end_index:
                planning = "No plans currently."
            else:
                planning = output[start_index:end_index].strip()

            planning = planning.replace('new', 'previous').replace('New', 'Previous')

            self.planning[i].append(planning)

        return planning

    def parse_json_planning(self, json_output):
        try:
            planning = json_output["New Planning"]
        except:
            planning = "No plans currently."

        self.planning[0].append(planning)
        return planning

    def parse_action(self, nav_output, only_options_batch, t):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        output_batch = []
        output_index_batch = []

        for i in range(batch_size):
            output = nav_output[i].strip()

            pattern = re.compile("Action")  # keyword
            matches = pattern.finditer(output)
            indices = [match.start() for match in matches]
            output = output[indices[-1]:]

            search_result = re.findall(r"Action:\s*([A-M])", output)
            if search_result:
                output = search_result[-1]

                if output in only_options_batch[i]:
                    output_batch.append(output)
                    output_index = only_options_batch[i].index(output)
                    output_index_batch.append(output_index)
                else:
                    output_index = 0
                    output_index_batch.append(output_index)
            else:
                output_index = 0
                output_index_batch.append(output_index)

        if bool(self.args.stop_after):
            if t < self.args.stop_after:
                for i in range(batch_size):
                    output_index_batch[i] = output_index_batch[i] + 1  # add 1 to index (avoid stop within 3 steps)
        return output_index_batch

    def parse_json_action(self, json_output, only_options_batch, t):
        try:
            output = str(json_output["Action"])
            if output in only_options_batch[0]:
                output_index = only_options_batch[0].index(output)
            else:
                output_index = 0

        except:
            output_index = 0

        if bool(self.args.stop_after):
            if t < self.args.stop_after:
                output_index += 1  # add 1 to index (avoid stop within 3 steps)

        output_index_batch = [output_index]
        return output_index_batch

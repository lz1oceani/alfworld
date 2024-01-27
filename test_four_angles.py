import sys
import os
from PIL import Image
import cv2
os.environ['DISPLAY'] = ":1"
os.environ['ALFRED_ROOT'] = "/home/ada/PycharmProjects/ALFRED/alfred"
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
from ai2thor.controller import Controller
from moviepy.editor import ImageSequenceClip
from typing import Sequence
import numpy as np
from pathlib import Path

from env.thor_env import ThorEnv
import json

def save_video(frames, save_path, fps=10):
    frames = ImageSequenceClip(frames, fps=fps)
    frames.write_videofile(save_path, fps=fps)


def is_json_file(filepath):
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True
    except ValueError:
        return False
    except FileNotFoundError:
        print("File not found")
        return False


def get_json_files(data_path):
    train_data_list = []
    for filepath,dirnames,filenames in os.walk(data_path):
        for filename in filenames:
            json_path = os.path.join(filepath,filename)
            if is_json_file(json_path):
                train_data_list.append(json_path)
            if len(train_data_list) == 200:
                return train_data_list
    return train_data_list


def get_parent_folder_name(file_path):
    return Path(file_path).parent.name


def save_as_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def split_and_rearrange(arr, element):
    # Check if element exists in the ndarray
    indices = np.where(arr == element)
    if len(indices[0]) == 0:
        return arr
    
    # Get the first index of the specific element
    index = indices[0][0]
    
    # Split and rearrange the ndarray
    return np.concatenate(([element], arr[index+1:], arr[:index]))


def generate_array_containing_number(number, total_points=80):
    if not 0 <= number <= 360:
        raise ValueError("The number must be between 1 and 360, but the number  is{}".format(number))
    
   # Create a linear space
    arr = np.linspace(0, 360, total_points).round().astype(int)

    # If the number is not in the array, replace the closest value with the number
    if number not in arr:
        idx = (np.abs(arr - number)).argmin()
        arr[idx] = number

    arr = split_and_rearrange(arr, number)
    
    return arr

def main(json_file_list, save_folder):
    env = ThorEnv(player_screen_width=2048,player_screen_height=2048)
    # load json data
    for i, json_file in enumerate(json_file_list):
        parent_folder_name = get_parent_folder_name(json_file)
        current_folder = save_folder + "/" + parent_folder_name
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        
        anns_json_path = current_folder + "/" + parent_folder_name + ".json"
        video_save_path = current_folder + "/" + parent_folder_name + ".mp4"
        
        with open(json_file) as f:
            traj_data = json.load(f)

            # setup
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            anns_dict = dict(anns=traj_data['turk_annotations']["anns"])

            scene_name = 'FloorPlan%d' % scene_num
            
            anns_dict['FloorPlan']=scene_num
    
            env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            env.step(dict(action='Initialize', gridSize=0.5, fieldOfView='110', visibilityDistance='4'))
            env.step(dict(traj_data['scene']['init_action']))
            
            
            position = env.last_event.metadata['agent']['position']
            rotation = env.last_event.metadata['agent']['rotation']
            
            degrees = generate_array_containing_number(rotation['y'], total_points=8)
            
            frames = []
            for degree in degrees:
                teleport_action = {
                    'action': 'TeleportFull',
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'rotation': dict(x=rotation['x'], y=degree, z=rotation['z'])
                }
                env.step(teleport_action)
                frames.append(env.last_event.frame)
                
            save_video(frames, video_save_path, fps=8)
            save_as_json(anns_dict, anns_json_path)
            print("Finish {}-th data".format(i+1))

    env.stop()


def check_img_quality(img_path):
    img = cv2.imread(img_path)
    print(img.shape)


if __name__ == "__main__":
    img_path = "/home/ada/PycharmProjects/ALFRED/alfred/gen/test_folder/test2048.jpg"
    data_path = "/home/ada/PycharmProjects/ALFRED/full_2.1.0/train"
    save_folder = "/home/ada/PycharmProjects/ALFRED/alfred/gen/test_folder/save"
    json_file_list = get_json_files(data_path)
    main(json_file_list, save_folder)
    #check_img_quality(img_path)
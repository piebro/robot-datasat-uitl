import sys
import json

import numpy as np
import robot_dataset_util as util
import render_robot_position
import robot_position

from field_dimension import FieldDimensions


def get_random_plausible_robot_position(verbose=0):
    # add the argument to specify how many pixel of each label is needed for the images to be plausible
    fd = FieldDimensions()

    x = np.random.randint(fd.border_strip_width, fd.size_x-fd.border_strip_width)
    y = np.random.randint(fd.mid_y, fd.size_y-fd.border_strip_width)
    rot_left_right_torso = np.random.randint(0, 360)
    rot_left_right_head = 0
    rot_top_down = np.random.randint(55, 90)
    rot_clockwise = 0
    pos = robot_position.RobotPosition(x, y, rot_left_right_torso, rot_left_right_head, rot_top_down, rot_clockwise)

    rendered_rgb_pos = render_robot_position.get_robot_position_rendered_rgb_image(pos, 110)
    class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
    rendered_label_mask = render_robot_position.rgb_mask_to_label_mask(rendered_rgb_pos, class_id_to_color)
    rendered_rgb_pos_from_mask = util.get_colored_segmentation_mask(rendered_label_mask, class_id_to_color)

    unique_label, label_count = np.unique(rendered_label_mask, return_counts=True)
    if verbose > 1:
        print(unique_label, label_count)

    if 1 in unique_label and label_count[1] > 100 and 2 in unique_label and label_count[2] > 100 and 3 in unique_label and label_count[3] > 100:
        if verbose > 0:
            util.get_figure_of_images([["rendered_rgb_pos_from_mask", rendered_rgb_pos_from_mask]], 1, 1, 10)
        return pos, rendered_label_mask
    else:
        return get_random_plausible_robot_position(verbose)

def get_random_plausible_robot_position_and_annotation(verbose=0):
    pos, rendered_label_mask = get_random_plausible_robot_position(verbose)
    anns = util.annotations_from_mask_semantic_segmentation(rendered_label_mask)
    return pos, anns
    
def save_syntetic_dataset(save_path, num_of_samples):
    pos_and_anns = []
    for i in range(num_of_samples):
        pos, anns = get_random_plausible_robot_position_and_annotation()
        pos = [pos.x, pos.y, pos.rot_left_right_torso, pos.rot_left_right_head, pos.rot_top_down, pos.rot_clockwise]
        pos_and_anns.append([pos, anns])
        
        sys.stdout.write("\r"+str(i+1)+" / "+str(num_of_samples))
        sys.stdout.flush()
    print("")

    with open(save_path, 'w') as f:   
        json.dump(pos_and_anns, f)

def view_position_and_mask_annotation_dataset(dataset_path, num_of_images, fig_size=8):
    with open(dataset_path) as f:
        pos_and_anns = np.array(json.load(f))
    
    selection = np.random.choice(range(len(pos_and_anns)), num_of_images)
    pos_and_anns = pos_and_anns[selection]
    
    img_array = []
    for pos, anns in pos_and_anns:
        mask = util.get_mask_segmentation(anns, 640, 480)
        class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
        mask_rgb = util.get_colored_segmentation_mask(mask, class_id_to_color)
        pos_str = f"x:{pos[0]}, y:{pos[1]}, rot_left_right_torso:{pos[2]}, rot_left_right_head:{pos[3]}, rot_top_down:{pos[4]}, rot_clockwise:{pos[5]}"
        img_array.append([pos_str, mask_rgb])
        img_array.append(["position", render_robot_position.get_robot_pos_drawing([robot_position.RobotPosition(*pos)],[[0,0,255]])])
    
    util.get_figure_of_images(img_array, 2, int(len(img_array)/2), fig_size)

def comibine_datasets(dataset_paths, save_path):
    pos_and_anns = []
    for dataset_path in dataset_paths:
        with open(dataset_path) as f:
            pos_and_anns.extend(json.load(f))
    
    with open(save_path, 'w') as f:
        json.dump(pos_and_anns, f)

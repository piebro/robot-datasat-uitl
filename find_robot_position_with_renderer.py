from timeit import default_timer
import os
import sys

import numpy as np
from matplotlib import cm as colormap
import pandas as pd

from field_dimension import FieldDimensions
from render_robot_position import get_robot_pos_drawing, get_robot_position_rendered_rgb_image, rgb_mask_to_label_mask, get_diff_visualisation
from robot_position import RobotPosition, add_robot_postions
import robot_dataset_util as util

import blackbox as bb

class CustomExecutor:
    
    def __init__(self):
        pass
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass
    
    @staticmethod
    def map(fun, params):
        results = []
        for param in params:
            results.append(fun(param))
        return results


def get_robot_pos_drawing_from_search_result(search_robot_positions_and_loss):
    robot_positions = []
    colors = []
    min_loss = np.min(search_robot_positions_and_loss[:,1])
    max_loss = np.max(search_robot_positions_and_loss[:,1])
    loss_delta = max_loss- min_loss
    search_robot_positions_and_loss = search_robot_positions_and_loss[::-1]
    for temp_robot_pos, temp_loss in search_robot_positions_and_loss:
        robot_positions.append(temp_robot_pos)
        rel_loss = ((temp_loss- min_loss)/loss_delta)
        cmap = colormap.get_cmap("viridis")
        color = (np.int32(np.array(cmap(rel_loss))[:-1]*255)).tolist()
        colors.append(color)
    return get_robot_pos_drawing(robot_positions, colors, circle_radius=4, arrow_thickness=2)

def get_robot_position_optimize_renderer(gt_label_mask, init_robot_position, domain, loss_func, num_of_optimizing_steps, batchsize, rendered_line_width=110, verbose=0):
    # restrict the domain if it is not in the field
    fd = FieldDimensions()
    if isinstance(domain["y"], list):
        domain["y"][1] = np.min([domain["y"][1], fd.border_strip_width + fd.field_y - init_robot_position.y])
    if isinstance(domain["x"], list):
        domain["x"][0] = np.max([domain["x"][0], fd.border_strip_width - init_robot_position.x])
        domain["x"][1] = np.min([domain["x"][1], fd.border_strip_width + fd.field_x - init_robot_position.x])

    # create the domain for the fitting function
    par_to_key = []
    domain_array = []
    robo_pos = {}
    for key, value in domain.items():
        if isinstance(value, list):
            par_to_key.append(key)
            domain_array.append(value)
        else:
            robo_pos[key] = value
            
    def function_to_optimize(par):
        for i, key in enumerate(par_to_key):
            robo_pos[key] = par[i]
        
        diff_robot_pos = RobotPosition(robo_pos["x"], robo_pos["y"], robo_pos["rot_left_right_torso"], robo_pos["rot_left_right_head"], robo_pos["rot_top_down"], robo_pos["rot_clockwise"])
        robot_pos = add_robot_postions(init_robot_position, diff_robot_pos)
        rendered_rgb_image = get_robot_position_rendered_rgb_image(robot_pos, rendered_line_width)
        

        class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
        rendered_label_mask = rgb_mask_to_label_mask(rendered_rgb_image, class_id_to_color)
        loss = loss_func(gt_label_mask, rendered_label_mask)
        return loss
    
    init_loss = function_to_optimize(np.zeros(len(domain)))
    if verbose > 1:
        print("init_robot_position loss:", init_loss)
    
    start = default_timer()
    bb.search_min(f=function_to_optimize, domain=domain_array, budget=num_of_optimizing_steps, batch=batchsize, resfile='output.csv', executor=CustomExecutor)
    end = default_timer()
    if verbose > 1:
        print("calculation with {} steps took {:.2f}s and {:.2f}s/step".format(num_of_optimizing_steps, end - start, (end - start)/num_of_optimizing_steps))

    search_result = pd.read_csv('output.csv').values
    os.remove('output.csv')
    
    search_robot_positions_and_loss = []
    for res_row in search_result:
        for i, key in enumerate(par_to_key):
            robo_pos[key] = res_row[i]
        diff_robot_pos = RobotPosition(robo_pos["x"], robo_pos["y"], robo_pos["rot_left_right_torso"], robo_pos["rot_left_right_head"], robo_pos["rot_top_down"], robo_pos["rot_clockwise"])
        temp_robot_pos = add_robot_postions(init_robot_position, diff_robot_pos)
        search_robot_positions_and_loss.append([temp_robot_pos, res_row[-1]])
        
    search_robot_positions_and_loss.append([init_robot_position, init_loss])
    search_robot_positions_and_loss = np.array(search_robot_positions_and_loss)
    search_robot_positions_and_loss = search_robot_positions_and_loss[np.argsort(search_robot_positions_and_loss[:,1])]
    
    if verbose > 1:
        print("best loss at:", search_robot_positions_and_loss[0,1])
    
    return search_robot_positions_and_loss

def get_robot_position_optimize_renderer_grid_search(gt_label_mask, domain, loss_func, rendered_line_width = 110, verbose=0):
    fd = FieldDimensions()
    init_robot_pos = RobotPosition(
        x=fd.border_strip_width + fd.field_x*2/4, y=fd.border_strip_width + fd.field_y*3/4, rot_left_right_torso=180, rot_left_right_head=0, rot_top_down=65, rot_clockwise=0)

    domain_values = []
    for _, value in domain.items():
        if isinstance(value, list):
            dx = (value[1]-value[0])/(value[2])
            domain_values.append(np.arange(value[0]+dx/2, value[1], dx))
        else:
            domain_values.append(np.array([value]))

    all_combinations = np.stack(np.meshgrid(*domain_values), -1).reshape(-1, len(domain_values))
    if verbose > 0:
        print("there are ", len(all_combinations), "combinations")
    
    results = []
    for i, combination in enumerate(all_combinations):
        diff_robot_pos = RobotPosition(*combination)
        robot_pos = add_robot_postions(init_robot_pos, diff_robot_pos)
        rendered_rgb_image = get_robot_position_rendered_rgb_image(robot_pos, rendered_line_width)
        class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
        rendered_label_mask = rgb_mask_to_label_mask(rendered_rgb_image, class_id_to_color)
        loss = loss_func(gt_label_mask, rendered_label_mask)

        results.append([robot_pos, loss])    
        if verbose > 0:
            sys.stdout.write("\r"+str(i+1)+" / "+str(len(all_combinations)))
            sys.stdout.flush()

    if verbose > 0:
        print("")
    results = np.array(results)
    results = results[np.argsort(results[:,1])]
    #search_robot_positions_and_loss = find_robot_pos_grid_search(gt_label_mask, verbose=1)
    best_robot_pos = results[0,0]
    best_loss = results[0,1]
    if verbose > 1:
      
        class_id_to_color = [[0, 0, 0], [255,0,0], [0,255,0], [0,0,255], [122,122,0], [0,122,122], [122,0,122], [255,122,122], [122,255,122], [122,122,255]]
        best_robot_view = get_robot_position_rendered_rgb_image(best_robot_pos, 110)
        best_robot_diff = get_diff_visualisation(gt_label_mask, rgb_mask_to_label_mask(best_robot_view, class_id_to_color))

        img_array = [
          ["cmask", util.get_colored_segmentation_mask(gt_label_mask, class_id_to_color)],
          ["cmask", util.get_colored_segmentation_mask(gt_label_mask, class_id_to_color)],
          ["search_space", get_robot_pos_drawing_from_search_result(results)],

          ["best_robot_view", best_robot_view],
          ["best_robot_diff", util.get_colored_segmentation_mask(best_robot_diff, class_id_to_color)],
          ["best_robot_top_view", get_robot_pos_drawing([best_robot_pos], [[0, 0, 255]])],
        ]
        util.get_figure_of_images(img_array, 3, 2, 7)
        print("best loss:", best_loss)
    return results

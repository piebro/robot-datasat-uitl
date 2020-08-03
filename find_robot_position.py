import robot_dataset_util as util
import robot_position
import render_robot_position
import find_robot_position_with_keypoints as frp_keypoints
import find_robot_position_with_renderer as frp_renderer
from field_dimension import FieldDimensions

def get_position_of_mask_with_keypoints_and_search(annotations, loss, num_of_optimizing_steps, batchsize, rendered_line_width, verbose=0):
    mask = util.get_mask_segmentation(annotations, 640, 480)

    init_robot_pos = frp_keypoints.get_robot_position_with_key_points(mask, verbose)

    if init_robot_pos is None:
        fd = FieldDimensions()
        init_robot_pos = robot_position.RobotPosition(
            x=fd.border_strip_width + fd.field_x*2/4, y=fd.border_strip_width + fd.field_y*3/4, rot_left_right_torso=180, rot_left_right_head=0, rot_top_down=65, rot_clockwise=0)
        domain = {
            "x": [-fd.field_x/2, fd.field_x/2],
            "y": [-fd.field_y/4, fd.field_y/4],
            "rot_left_right_torso":[-180, 180],
            "rot_left_right_head":0,
            "rot_top_down":[-25, 25],
            "rot_clockwise":0
        }
    else:
        domain = {
            "x": [-1000, 1000],
            "y": [-1000, 1000],
            "rot_left_right_torso":[-30, 30],
            "rot_left_right_head":0,
            "rot_top_down":[-15, 15],
            "rot_clockwise":0
        }
      
    search_robot_positions_and_loss = frp_renderer.get_robot_position_optimize_renderer(
        mask, init_robot_pos, domain, loss, num_of_optimizing_steps, batchsize, rendered_line_width, verbose)

    best_robot_pos = search_robot_positions_and_loss[0,0]
    bost_loss = search_robot_positions_and_loss[0,1]

    if verbose > 0:
        class_id_to_color = [[0, 0, 0], [255,0,0], [0,255,0], [0,0,255], [122,122,0], [0,122,122], [122,0,122], [255,122,122], [122,255,122], [122,122,255]]

        init_robot_view = render_robot_position.get_robot_position_rendered_rgb_image(init_robot_pos, 110)
        init_robot_diff = render_robot_position.get_diff_visualisation(mask, render_robot_position.rgb_mask_to_label_mask(init_robot_view, class_id_to_color))

        best_robot_view = render_robot_position.get_robot_position_rendered_rgb_image(best_robot_pos, 110)
        best_robot_diff = render_robot_position.get_diff_visualisation(mask, render_robot_position.rgb_mask_to_label_mask(best_robot_view, class_id_to_color))

        img_array = [
            ["cmask", util.get_colored_segmentation_mask(mask, class_id_to_color)],
            ["cmask", util.get_colored_segmentation_mask(mask, class_id_to_color)],
            ["search_space", frp_renderer.get_robot_pos_drawing_from_search_result(search_robot_positions_and_loss)],

            ["init_robot_view", init_robot_view],
            ["init_robot_diff", util.get_colored_segmentation_mask(init_robot_diff, class_id_to_color)],
            ["init_robot_top_view", render_robot_position.get_robot_pos_drawing([init_robot_pos], [[0, 0, 255]])],

            ["best_robot_view", best_robot_view],
            ["best_robot_diff", util.get_colored_segmentation_mask(best_robot_diff, class_id_to_color)],
            ["best_robot_top_view", render_robot_position.get_robot_pos_drawing([best_robot_pos], [[0, 0, 255]])],
        ]
        fig = util.get_figure_of_images(img_array, 3, 3, 10)
        return best_robot_pos, bost_loss, fig

    return best_robot_pos, bost_loss
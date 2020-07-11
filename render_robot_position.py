import numpy as np
import pyrender
import trimesh
import math
import cv2

from field_dimension import FieldDimensions
from find_robot_position_with_keypoints import draw_field_key_points, get_key_points_top_down_field

def get_diff_visualisation(gt_label_mask, pred_label_mask):
    diff_mask = np.copy(pred_label_mask)
    for label in range(1,5):#np.unique(pred_label_mask):
        if label == 0:
            continue
        gt_covered = (gt_label_mask==label)
        #gt_covered_or_robot_or_ball = np.logical_or(gt_covered, gt_label_mask==5)
        pred_covered = (pred_label_mask==label)
        
        error_in_gt_not_in_pred = np.logical_and(gt_covered, np.logical_not(pred_covered))#np.logical_or(, )
        diff_mask[error_in_gt_not_in_pred] = 5
        
        error_in_pred_not_in_gt = np.logical_and(pred_covered, np.logical_not(gt_covered))
        diff_mask[error_in_pred_not_in_gt] = 6
    return diff_mask

def rgb_mask_to_label_mask(rgb_mask, class_id_to_color):
    label_mask = np.zeros(rgb_mask.shape[:2])
    for label, color in enumerate(class_id_to_color):
        if label == 0:
            continue
        label_mask[np.all(rgb_mask == color, axis=-1)] = label
    return np.int8(label_mask)

def draw_circle(scene, center, radius, color, thickness):
    circle_trimesh = trimesh.creation.annulus(radius, radius+thickness, 0.1)
    circle_trimesh.visual.face_colors = np.repeat(np.array(color)[np.newaxis, :], circle_trimesh.faces.shape[0], axis=0)
    circle_mesh = pyrender.Mesh.from_trimesh(circle_trimesh, smooth=False)
    circle_node = pyrender.Node(mesh=circle_mesh, translation=np.array([center[0], center[1], 2]))
    scene.add_node(circle_node)

def draw_line(scene, center, length, left_right, color, thickness):
    if left_right:
        draw_box(scene, [center[0], center[1], -2], [length, thickness, 0.1], color)
    else:
        draw_box(scene, [center[0], center[1], -2], [thickness, length, 0.1], color)

    
def draw_box(scene, center, extends, color):
    box_trimesh = trimesh.creation.box(extents=extends)
    box_trimesh.visual.face_colors = np.repeat(np.array(color)[np.newaxis, :], box_trimesh.faces.shape[0], axis=0)
    box_mesh = pyrender.Mesh.from_trimesh(box_trimesh, smooth=False)
    box_node = pyrender.Node(mesh=box_mesh, translation=np.array([center[0], center[1], center[2]]))
    scene.add_node(box_node)

def render_scene(scene, width, height):
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    render_img, _ = r.render(scene)
    r.delete()
    return render_img

def translate(camera_pose, x, y, z):
    camera_pose[0,3] += x
    camera_pose[1,3] += y
    camera_pose[2,3] += z
    
def rotate_x_axis(camera_pose, angle):
    rotMat = np.eye(4)
    rotMat[1,1] = math.cos(angle)
    rotMat[1,2] = -math.sin(angle)
    rotMat[2,1] = math.sin(angle)
    rotMat[2,2] = math.cos(angle)
    return np.matmul(camera_pose,rotMat)#camera_pose*a

def rotate_y_axis(camera_pose, angle):
    rotMat = np.eye(4)
    rotMat[0,0] = math.cos(angle)
    rotMat[0,2] = math.sin(angle)
    rotMat[2,0] = -math.sin(angle)
    rotMat[2,2] = math.cos(angle)
    return np.matmul(camera_pose,rotMat)

def rotate_z_axis(camera_pose, angle):
    rotMat = np.eye(4)
    rotMat[0,0] = math.cos(angle)
    rotMat[0,1] = -math.sin(angle)
    rotMat[1,0] = math.sin(angle)
    rotMat[1,1] = math.cos(angle)
    return np.matmul(camera_pose, rotMat)


def get_robot_pos_drawing(robot_positions, colors, circle_radius=10, arrow_thickness=4):
    field_drawing, field_drawing_scale = draw_field_key_points([], 500, 3, 10, get_key_points_top_down_field(FieldDimensions()), [], draw_key_points=False)
    if robot_positions is None or colors is None:
        return field_drawing
    
    for robot_pos, color in zip(robot_positions, colors):
        tx = robot_pos.x*field_drawing_scale
        ty = robot_pos.y*field_drawing_scale
        cv2.circle(field_drawing, (int(tx), int(ty)), circle_radius, color, -1)
        look_at = [math.cos((-robot_pos.rot_left_right_torso+90)/180*np.pi)*30+tx, math.sin((-robot_pos.rot_left_right_torso+90)/180*np.pi)*30+ty]
        cv2.arrowedLine(field_drawing, (int(tx), int(ty)), (int(look_at[0]), int(look_at[1])), color, thickness=arrow_thickness, tipLength=0.4)
    return field_drawing

def add_camera(scene, x, y, z, rot_torso, rot_x, rot_y, rot_z):
    camera_pose = np.eye(4)
    
    camera_pose = rotate_x_axis(camera_pose, np.pi/180 * 90)
    camera_pose = rotate_y_axis(camera_pose, np.pi/180 * rot_torso)
    camera_pose = rotate_x_axis(camera_pose, np.pi/180 * -90)
    
    camera_pose = rotate_x_axis(camera_pose, np.pi/180 * rot_x)
    camera_pose = rotate_y_axis(camera_pose, np.pi/180 * rot_y)
    camera_pose = rotate_z_axis(camera_pose, np.pi/180 * rot_z)
    
    translate(camera_pose, x, y, z)
    
    #67.4°  DFOV (56.3° HFOV, 43.7°  VFOV)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/180 * 43.7, znear=500)
    scene.add(camera, pose=camera_pose)

def get_robot_position_rendered_rgb_image(robot_pos, line_width):
    robot_camera_height = 45.19 + 102.90 + 100 + 85 + 126.50 + 63.64
    scene = pyrender.Scene(ambient_light=[0.99999, 0.99999, 0.99999], bg_color=[0, 0, 0])
    size_x, _ = draw_on_scene(scene, line_width)
    add_camera(scene, size_x-robot_pos.x, robot_pos.y, robot_camera_height, robot_pos.rot_left_right_torso, robot_pos.rot_top_down, robot_pos.rot_left_right_head, robot_pos.rot_clockwise)
    rendered_rgb_mask = render_scene(scene, 640, 480)
    return rendered_rgb_mask


def draw_on_scene(scene, line_thickness):
    fd = FieldDimensions()
    
    class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
    line_color = class_id_to_color[1]
    center_circle_color = class_id_to_color[2]
    goal_color = class_id_to_color[3]
    penalty_cross_color = class_id_to_color[4]
    
    draw_line(scene, [fd.mid_x,fd.mid_y], fd.field_x, True, line_color, line_thickness)
    
    draw_line(scene, [fd.border_strip_width, fd.mid_y], fd.field_y, False, line_color, line_thickness)
    draw_line(scene, [fd.size_x-fd.border_strip_width, fd.mid_y], fd.field_y, False, line_color, line_thickness)
    
    draw_line(scene, [fd.mid_x, fd.border_strip_width], fd.field_x+line_thickness, True, line_color, line_thickness)
    draw_line(scene, [fd.mid_x, fd.size_y-fd.border_strip_width], fd.field_x+line_thickness, True, line_color, line_thickness)
    
    draw_line(scene, [fd.mid_x, fd.border_strip_width+fd.penalty_area_length], fd.penalty_area_width+line_thickness, True, line_color, line_thickness)
    draw_line(scene, [fd.mid_x-fd.penalty_area_width/2, fd.border_strip_width+fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    draw_line(scene, [fd.mid_x+fd.penalty_area_width/2, fd.border_strip_width+fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    
    draw_line(scene, [fd.mid_x, fd.size_y-fd.border_strip_width-fd.penalty_area_length], fd.penalty_area_width+line_thickness, True, line_color, line_thickness)
    draw_line(scene, [fd.mid_x-fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    draw_line(scene, [fd.mid_x+fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    
    draw_circle(scene, [fd.mid_x, fd.mid_y], fd.center_circle_diameter/2, center_circle_color, line_thickness)
    
    draw_box(scene, [fd.mid_x, fd.border_strip_width, fd.goal_height/2+fd.goal_post_diameter/2], [fd.goal_width+fd.goal_post_diameter, 0.1, fd.goal_height+fd.goal_post_diameter], goal_color)
    draw_box(scene, [fd.mid_x, fd.size_y-fd.border_strip_width, fd.goal_height/2+fd.goal_post_diameter/2], [fd.goal_width+fd.goal_post_diameter, 0.1, fd.goal_height+fd.goal_post_diameter], goal_color)
    
    draw_line(scene, [fd.mid_x-fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    draw_line(scene, [fd.mid_x+fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length/2], fd.penalty_area_length, False, line_color, line_thickness)
    
    draw_line(scene, [fd.mid_x, fd.size_y-fd.border_strip_width-fd.penalty_cross_distance], line_thickness*1.25, False, line_color, line_thickness*0.70)
    draw_line(scene, [fd.mid_x, fd.size_y-fd.border_strip_width-fd.penalty_cross_distance], line_thickness*1.25, True, line_color, line_thickness*0.70)
    
    draw_line(scene, [fd.mid_x, fd.border_strip_width+fd.penalty_cross_distance], line_thickness*1.25, False, line_color, line_thickness*0.70)
    draw_line(scene, [fd.mid_x, fd.border_strip_width+fd.penalty_cross_distance], line_thickness*1.25, True, line_color, line_thickness*0.70)
    
    return [fd.size_x, fd.size_y]
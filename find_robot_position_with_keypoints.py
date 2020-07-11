import cv2
import numpy as np
import math

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from field_dimension import FieldDimensions
from robot_position import RobotPosition, add_robot_postions

def get_middle_pos(start, end):
    return start + (end-start)/2

def draw_scan_line_points(mask, circle_points, line_points):
    for circle_point in circle_points:
        cv2.circle(mask, (int(circle_point[0]), int(circle_point[1])), 1, [100, 255, 255], -1)

    for line_point in line_points:
        cv2.circle(mask, (int(line_point[0]), int(line_point[1])), 1, [0, 0, 255], -1)

def line_fit(line_points):
    x = line_points[:, 0]
    y = line_points[:, 1]

    m, b = np.polyfit(x, y, 1)
    min_x = np.min(x)
    max_x = np.max(x)
    status = True
    return status, m, b, min_x, m*min_x + b, max_x, m*max_x + b

def ran_color(min_val=0):
    return [
        np.random.randint(min_val, 255),
        np.random.randint(min_val, 255),
        np.random.randint(min_val, 255)
    ]

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_rotated_rect_points(rotated_rect):
    center_x, center_y = rotated_rect[0]
    width, height = rotated_rect[1]
    wh = width / 2
    hh = height / 2
    angle = rotated_rect[2]/180*np.pi
    points = [
        rotate([center_x, center_y], [center_x-wh, center_y-hh], angle),
        rotate([center_x, center_y], [center_x+wh, center_y-hh], angle),
        rotate([center_x, center_y], [center_x+wh, center_y+hh], angle),
        rotate([center_x, center_y], [center_x-wh, center_y+hh], angle)
    ]
    return np.array(points)

def get_index_middle_line(ellipse_center, lines):
    distances = []
    for i in range(len(lines)):
        m, b, _, _, _, _ = lines[i]
        distances.append(ellipse_center[1] - (ellipse_center[0]*m + b))
    min_index = np.argmin(distances)
    if distances[min_index] < 20:
        return min_index
    else:
        return None
        
def draw_key_points_on_img(img, kp_img_list, kp_radius, colors):
    for kp, color in zip(kp_img_list, colors):
        cv2.circle(img, tuple(np.int32(kp)), kp_radius, color, -1)

def get_keypoint_list(kp_mask, kp_field):
    kp_mask_list = []
    kp_field_list = []

    for key, _ in kp_mask.items():
        kp_mask_list.append(kp_mask[key])
        kp_field_list.append(kp_field[key])

    return np.array(kp_mask_list), np.array(kp_field_list)

def draw_line_cluster(img, line_cluster, draw_points=True, thickness=2):
    for line_points in line_cluster:
        color = ran_color(100)
        color_shift = 60
        dark_color = [max(color[0]-color_shift, 0), max(color[1]-color_shift, 0), max(color[2]-color_shift, 0)]
        if draw_points:
            for point in line_points:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, dark_color, -1)

        status, m, b, x1, y1, x2, y2 = line_fit(line_points)
        if status:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img

def get_key_points_top_down_field(field_dimensions):
    fd = field_dimensions

    color = (200, 200, 200)
    thickness = fd.line_width

    kp = {}
    kp["middle"] = [fd.mid_x, fd.mid_y]
    kp["field_middle_left"] = [fd.mid_x-fd.field_x/2, fd.mid_y]
    kp["field_middle_right"] = [fd.mid_x+fd.field_x/2, fd.mid_y]

    r = fd.center_circle_diameter/2
    kp["centercircle_left"] = [fd.mid_x-r, fd.mid_y]
    kp["centercircle_right"] = [fd.mid_x+r, fd.mid_y]
    kp["centercircle_top"] = [fd.mid_x, fd.mid_y-r]
    kp["centercircle_bottom"] = [fd.mid_x, fd.mid_y+r]
    kp["centercircle_top_right"] =\
      [fd.mid_x + r*math.cos(-np.pi/4), fd.mid_y + r*math.sin(-np.pi/4)]
    kp["centercircle_top_left"] =\
      [fd.mid_x + r*math.cos(-3*np.pi/4), fd.mid_y + r*math.sin(-3*np.pi/4)]
    kp["centercircle_bottom_left"] =\
      [fd.mid_x + r*math.cos(-5*np.pi/4), fd.mid_y + r*math.sin(-5*np.pi/4)]
    kp["centercircle_bottom_right"] =\
      [fd.mid_x + r*math.cos(-7*np.pi/4), fd.mid_y + r*math.sin(-7*np.pi/4)]

    kp["field_top_left"] = [fd.mid_x-fd.field_x/2, fd.mid_y-fd.field_y/2]
    kp["field_top_right"] = [fd.mid_x+fd.field_x/2, fd.mid_y-fd.field_y/2]
    kp["field_bottom_right"] = [fd.mid_x+fd.field_x/2, fd.mid_y+fd.field_y/2]
    kp["field_bottom_left"] = [fd.mid_x-fd.field_x/2, fd.mid_y+fd.field_y/2]

    kp["penalty_top_area_top_left"] = [fd.mid_x-fd.penalty_area_width/2, fd.border_strip_width]
    kp["penalty_top_area_top_right"] = [fd.mid_x+fd.penalty_area_width/2, fd.border_strip_width]
    kp["penalty_top_area_bottom_right"] =\
      [fd.mid_x+fd.penalty_area_width/2, fd.border_strip_width+fd.penalty_area_length]
    kp["penalty_top_area_bottom_left"] =\
      [fd.mid_x-fd.penalty_area_width/2, fd.border_strip_width+fd.penalty_area_length]

    kp["penalty_bottom_area_top_left"] =\
      [fd.mid_x-fd.penalty_area_width/2, fd.size_y-fd.border_strip_width]
    kp["penalty_bottom_area_top_right"] =\
      [fd.mid_x+fd.penalty_area_width/2, fd.size_y-fd.border_strip_width]
    kp["penalty_bottom_area_bottom_right"] =\
      [fd.mid_x+fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length]
    kp["penalty_bottom_area_bottom_left"] =\
      [fd.mid_x-fd.penalty_area_width/2, fd.size_y-fd.border_strip_width-fd.penalty_area_length]

    kp["penalty_cross_top"] = [fd.mid_x, fd.border_strip_width + fd.penalty_cross_distance]
    kp["penalty_cross_bottom"] =\
      [fd.mid_x, fd.size_y - fd.border_strip_width - fd.penalty_cross_distance]

    kp["goal_top_left"] = [fd.mid_x-fd.goal_width/2, fd.border_strip_width]
    kp["goal_top_right"] = [fd.mid_x+fd.goal_width/2, fd.border_strip_width]
    kp["goal_bottom_left"] = [fd.mid_x-fd.goal_width/2, fd.size_y - fd.border_strip_width]
    kp["goal_bottom_right"] = [fd.mid_x+fd.goal_width/2, fd.size_y - fd.border_strip_width]

    for key, val in kp.items():
        kp[key] = np.array(val)

    return kp

def draw_field_key_points(
  kp_field_list, size_x, line_width, kp_radius, kp_field, colors, draw_key_points=True):
    color = (200, 200, 200)
    thickness = line_width

    scale = size_x/(kp_field["middle"][0]*2)
    skp = {}
    for key, val in kp_field.items():
        t = np.int32(val*scale)
        skp[key] = (t[0], t[1])

    f = np.zeros((skp["middle"][1]*2, skp["middle"][0]*2, 3), dtype=np.uint8)

    # mid circle
    radius = skp["middle"][0]-skp["centercircle_left"][0]
    cv2.circle(f, skp["middle"], int(radius), color, thickness)

    # mid line
    cv2.line(f, skp["field_middle_left"], skp["field_middle_right"], color, thickness)

    # field border
    field_border_points = np.int32([[skp["field_top_left"], skp["field_top_right"],
                                     skp["field_bottom_right"], skp["field_bottom_left"]]])
    cv2.polylines(f, field_border_points, 1, color, thickness)

    # penalty top area
    cv2.polylines(f, np.int32([[skp["penalty_top_area_top_left"], skp["penalty_top_area_top_right"], skp["penalty_top_area_bottom_right"], skp["penalty_top_area_bottom_left"]]]), 1, color, thickness)
    cv2.polylines(f, np.int32([[skp["penalty_bottom_area_top_left"], skp["penalty_bottom_area_top_right"], skp["penalty_bottom_area_bottom_right"], skp["penalty_bottom_area_bottom_left"]]]), 1, color, thickness)

    # penalty cross
    cv2.circle(f, skp["penalty_cross_top"], thickness, color, thickness, -1)
    cv2.circle(f, skp["penalty_cross_bottom"], thickness, color, thickness, -1)

    if draw_key_points:
        for key, val in skp.items():
            cv2.circle(f, val, kp_radius, [255, 0, 0], -1)

        for kp, color in zip(kp_field_list, colors):
            cv2.circle(f, (int(kp[0]*scale), int(kp[1]*scale)), kp_radius, color, -1)

    return f, scale

def get_pixel_with_label(mask, label, swap_xy=True):
    if swap_xy:
        return np.swapaxes(np.array(np.where(mask == label)), 0, 1)[:, [1, 0]]
    else:
        return np.swapaxes(np.array(np.where(mask == label)), 0, 1)

def scan_line_points_top_bottom(points, ignore_line_at_edge, only_double_line):
    return scan_line_points_left_right(points[:, [1, 0]], ignore_line_at_edge, only_double_line)[:, [1, 0]]

def scan_line_points_left_right(points, ignore_line_at_edge, only_double_line):
    relevant_lines = np.unique(points[:, 1])
    new_points = []
    for x in relevant_lines:
        rel_pixel = points[points[:, 1] == x][:, 0]
        pixel_before_end_of_cluster = np.where(np.diff(rel_pixel, axis=0) != 1)[0]
        if len(pixel_before_end_of_cluster) == 0:
            if only_double_line:
                continue
            if ignore_line_at_edge:
                if rel_pixel[0] != 0 and rel_pixel[-1] != 480-1:
                    new_points.append([get_middle_pos(rel_pixel[0], rel_pixel[-1]), x])
            else:
                new_points.append([get_middle_pos(rel_pixel[0], rel_pixel[-1]), x])
        else:
            if rel_pixel[0] != 0:
                new_points.append([get_middle_pos(rel_pixel[0], rel_pixel[pixel_before_end_of_cluster[0]]), x])

            for i in range(1, len(pixel_before_end_of_cluster)):
                start_index = pixel_before_end_of_cluster[i-1] + 1
                end_index = pixel_before_end_of_cluster[i]
                new_points.append([get_middle_pos(rel_pixel[start_index], rel_pixel[end_index]), x])

            if rel_pixel[-1] != 480-1:
                new_points.append([get_middle_pos(rel_pixel[-1], rel_pixel[pixel_before_end_of_cluster[-1]+1]), x])

    return np.array(new_points)

def draw_points_on_mask(mask, points, radius, color):
    for p in points:
        cv2.circle(mask, (int(p[0]), int(p[1])), radius, color, -1)
    return mask


def cluster_line_points_dbscan(line_points, eps):
    line_points = np.array(line_points)
    X = StandardScaler().fit_transform(line_points)
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_

    line_cluster = []
    for label in np.unique(labels):
        cluster_lines_points = line_points[labels == label]
        cluster_lines_points = cluster_lines_points[np.lexsort((cluster_lines_points[:,1],cluster_lines_points[:, 0]))]
        line_cluster.append(cluster_lines_points)

    return line_cluster

def split_up_cluster(line_cluster, split_all_n_points):
    new_line_cluster = []
    for line_points in line_cluster:
        num_of_splits = max(1, int(len(line_points)/split_all_n_points))
        for split in np.array_split(line_points, num_of_splits):
            new_line_cluster.append(split)
    return new_line_cluster

def remove_lines_with_few_points(line_cluster, min_count_points):
    new_cluster = []
    for line_points in line_cluster:
        if len(line_points)>min_count_points:
            new_cluster.append(line_points)
    return new_cluster

def get_ellipse(mask, show=False):
    cmask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    circle_points = get_pixel_with_label(mask, 2)

    if len(circle_points) < 30:
        return None

    rotated_rect_ellipse = cv2.fitEllipse(np.int32(circle_points))
    if show:
        fig = util.get_figure_of_images([["circle_points_scan_line", draw_points_on_mask(cmask, circle_points, 2, [255,255,0])]], 1, 1, 10)

    return rotated_rect_ellipse

def extend_line(x1, y1, x2, y2, extend_mult):
    diff_x = x1-x2
    diff_y = y1-y2
    return [x1+extend_mult*diff_x, y1+extend_mult*diff_y, x2-extend_mult*diff_x, y2-extend_mult*diff_y]

def get_point_on_ellipse(center, radius, rotate_ellipse_angle, angle_on_ellipse):
    px = center[0]+radius[0]*math.cos(angle_on_ellipse)
    py = center[1]+radius[1]*math.sin(angle_on_ellipse)
    px, py = rotate(center, [px, py], rotate_ellipse_angle)
    return px, py

def get_intersection_ellipse_line_from_lines(rotated_rect_ellipse, line_cluster):
    for i, line_points in enumerate(line_cluster):
        status, m, b, x1, y1, x2, y2 = line_fit(line_points)
        points = get_intersection_ellipse_line_from_line(rotated_rect_ellipse, np.array([x1,y1]), np.array([x2,y2]), n=500)
        if points is not None:
            return points, i

    return None, None

def get_intersection_ellipse_line_from_line(rotated_rect_ellipse, p1, p2, n=200):
    center = rotated_rect_ellipse[0]
    radius = [rotated_rect_ellipse[1][0]/2, rotated_rect_ellipse[1][1]/2]
    rotate_ellipse_angle = rotated_rect_ellipse[2]/180*np.pi

    distances = []
    for a in range(n):
        angle_on_ellipse = np.pi*2/n*a
        px, py = get_point_on_ellipse(center, radius, rotate_ellipse_angle, angle_on_ellipse)
        dist = np.linalg.norm(np.cross(p2-p1, p1-np.array([px,py])))/np.linalg.norm(p2-p1)
        distances.append([dist, px, py, angle_on_ellipse])

    distances = np.array(distances)
    distances = distances[distances[:,0].argsort()]

    if distances[1,0] < 10:
        intersection_1, intersection_2 = distances[:2]
        if intersection_1[1] > intersection_2[1]:
            intersection_1, intersection_2 = intersection_2, intersection_1

        angle_left = intersection_1[3]
        angle_right = intersection_2[3]
        angle_diff_top = angle_right-angle_left
        angle_diff_bottom = np.pi*2-angle_diff_top

        points = {}
        points["centercircle_left"] = intersection_1[1:3]
        points["centercircle_right"] = intersection_2[1:3]

        points["centercircle_top_left"] =\
          get_point_on_ellipse(center, radius, rotate_ellipse_angle, angle_left+1*angle_diff_top/4)
        points["centercircle_top"] =\
          get_point_on_ellipse(center, radius, rotate_ellipse_angle, angle_left+2*angle_diff_top/4)
        points["centercircle_top_right"] =\
          get_point_on_ellipse(center, radius, rotate_ellipse_angle, angle_left+3*angle_diff_top/4)

        points["centercircle_bottom_left"] = get_point_on_ellipse(
            center, radius, rotate_ellipse_angle, angle_left-1*angle_diff_bottom/4)
        points["centercircle_bottom"] = get_point_on_ellipse(
            center, radius, rotate_ellipse_angle, angle_left-2*angle_diff_bottom/4)
        points["centercircle_bottom_right"] = get_point_on_ellipse(
            center, radius, rotate_ellipse_angle, angle_left-3*angle_diff_bottom/4)

        return points
    else:
        return None

def get_penalty_cross_center(mask):
    pixel = get_pixel_with_label(mask, 4)
    if len(pixel) > 0:
        return np.mean(pixel, axis=0)
    else:
        return None

def get_goal_bbox(mask):
    pixel = get_pixel_with_label(mask, 3)
    if len(pixel) > 50:
        min_xy = np.min(pixel, axis=0)
        max_xy = np.max(pixel, axis=0)
        return [[min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]], [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]]]
    else:
        return None

def get_concept_img(cmask, rotated_rect_ellipse, line_cluster, penalty_cross_center, goal_bbox):
    mask_concepts = np.zeros_like(cmask)
    draw_line_cluster(mask_concepts, line_cluster, draw_points=False, thickness=5)
    cv2.ellipse(mask_concepts, rotated_rect_ellipse, ran_color(), 5)
    if penalty_cross_center is not None:
        cv2.circle(mask_concepts, (int(penalty_cross_center[0]), int(penalty_cross_center[1])), 5, ran_color(), -1)

    if goal_bbox is not None:
        cv2.polylines(mask_concepts, np.int32([goal_bbox]), 1, (50,100,255), 4)

    return mask_concepts

def get_line_cluster(mask, show=False):
    cmask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    line_points = get_pixel_with_label(mask, 1)
    if len(line_points)<10:
        return None
    line_points_scan_line = scan_line_points_top_bottom(line_points, ignore_line_at_edge=False, only_double_line=False)
    
    line_cluster_dbscan = cluster_line_points_dbscan(line_points_scan_line, eps=0.04)
    #line_cluster_dbscan_split = split_up_cluster(line_cluster_dbscan, split_all_n_points=5000)
    lines_mb_clusterd = cluster_bdscan_lines_mb(line_cluster_dbscan, eps=0.8, only_use_mb=False)
    if lines_mb_clusterd is None:
        return None
    lines_mb_clusterd2 = cluster_bdscan_lines_mb(lines_mb_clusterd, eps=0.18, only_use_mb=True)
    if lines_mb_clusterd2 is None:
        return None
    lines_mb_clusterd_filtered = remove_lines_with_few_points(lines_mb_clusterd2, min_count_points=40)

    if show:
        img_array = [
            ["scanline_mask", draw_points_on_mask(np.copy(cmask), line_points_scan_line, 2, [0, 0, 255])],
            ["scanline_mask", draw_points_on_mask(np.copy(cmask), line_points_scan_line, 2, [0, 0, 255])],
            ["line_cluster_dbscan", draw_line_cluster(np.copy(cmask), line_cluster_dbscan)],
            #["line_cluster_dbscan_split", draw_line_cluster(np.copy(cmask), line_cluster_dbscan_split)],
            ["lines_mb_clusterd", draw_line_cluster(np.copy(cmask), lines_mb_clusterd)],
            ["lines_mb_clusterd", draw_line_cluster(np.copy(cmask), lines_mb_clusterd2)],
            ["lines_mb_clusterd_filtered", draw_line_cluster(np.copy(cmask), lines_mb_clusterd_filtered)],
        ]
        fig = util.get_figure_of_images(img_array, 3, 2, 10)

    return lines_mb_clusterd_filtered


def cluster_bdscan_lines_mb(line_cluster, eps, only_use_mb=True):
    line_cluster = np.array(line_cluster)
    lines_mb = []
    new_line_cluster = []
    for line_points in line_cluster:
        if len(line_points) < 2:
            continue
        new_line_cluster.append(line_points)
        status, m, b, x1, y1, x2, y2 = line_fit(line_points)
        if only_use_mb:
            lines_mb.append([m, b])
        else:
            lines_mb.append([m, b, x1, y1, x2, y2])

    line_cluster = np.array(new_line_cluster)

    lines_mb = np.array(lines_mb)
    if len(lines_mb)<1:
        return None

    X = StandardScaler().fit_transform(lines_mb)
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_

    new_cluster = []
    for label in np.unique(labels):
        new_cluster_label_flatten = []
        for part in line_cluster[labels == label]:
            new_cluster_label_flatten.extend(part)

        new_cluster_label_flatten = np.array(new_cluster_label_flatten)
        new_cluster.append(new_cluster_label_flatten)
    return new_cluster

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])

    
def line_intersection_on_line(line1, line2, eps):
    x11, y11, x12, y12 = line1[3:]
    x21, y21, x22, y22 = line2[3:]
    
    intersection = line_intersection([[x11,y11], [x12,y12]], [[x21,y21], [x22,y22]])
    
    if (intersection[0]<max(x11,x12)+eps and intersection[0]>min(x11,x12)-eps and
        intersection[0]<max(x21,x22)+eps and intersection[0]>min(x21,x22)-eps):
        return intersection
    else:
        return None

def get_intersection_lines_to_line_indices(line_index, lines, eps, ignore_lines_indices = []):
    intersections = []
    for i in range(len(lines)):
        if i != line_index and i not in ignore_lines_indices:
            intersection = line_intersection_on_line(lines[i], lines[line_index], eps)
            if intersection is not None:
                intersections.append(i)
    return intersections

def get_key_points_mask(rotated_rect_ellipse, line_cluster, penalty_cross_center, goal_bbox):
    
    if rotated_rect_ellipse is None or line_cluster is None:
        return []
    
    ellipse_points, middle_line_index = get_intersection_ellipse_line_from_lines(rotated_rect_ellipse, line_cluster)
    
    if ellipse_points is None:
        return []
    
    kp_mask = ellipse_points
    line_cluster_without_middle_line = line_cluster[:middle_line_index] + line_cluster[middle_line_index+1:]
    middle_line = line_fit(line_cluster[middle_line_index])
    
    if penalty_cross_center is not None:
        if penalty_cross_center[1] < kp_mask["centercircle_bottom"][1]:
            kp_mask["penalty_cross_top"] = penalty_cross_center
        else:
            kp_mask["penalty_cross_bottom"] = penalty_cross_center
        
    if goal_bbox is not None:
        if goal_bbox[3][0] != 0: # not at the edge of the image
            kp_mask["goal_top_left"] = goal_bbox[3]
        if goal_bbox[2][0] != 640-1: # not at the edge of the image
            kp_mask["goal_top_right"] = goal_bbox[2]
    
    return kp_mask


def get_robot_position_from_homography_matrix(h):
    viewport_height = 0.25
    view_points_in_mask = np.array([[0,480-(480*viewport_height)], [0,480], [640, 480], [640,480-(480*viewport_height)]])
    view_points_in_field_in_mm = cv2.perspectiveTransform(np.array([np.float32(view_points_in_mask)]), h)[0]
    robot_pos_in_mm = line_intersection(view_points_in_field_in_mm[0:2], view_points_in_field_in_mm[2:4])
    
    look_at_point_in_mm = view_points_in_field_in_mm[0] + (-view_points_in_field_in_mm[0] + view_points_in_field_in_mm[3])/2
    angle_left_right_torso = angle_between_2_points(robot_pos_in_mm, look_at_point_in_mm)

    robot_camera_height = 45.19 + 102.90 + 100 + 85 + 126.50 + 63.64
    dist_to_look_at_point = np.linalg.norm(robot_pos_in_mm-look_at_point_in_mm)
    angle_top_down = angle_between_2_points([0,dist_to_look_at_point/viewport_height], [robot_camera_height, 0])

    pos = RobotPosition(robot_pos_in_mm[0], robot_pos_in_mm[1], -angle_left_right_torso+90, 0, -angle_top_down, 0)
    
    return pos, view_points_in_mask, view_points_in_field_in_mm, look_at_point_in_mm


def get_robot_position_with_key_points(mask, verbose):
    rotated_rect_ellipse = get_ellipse(mask, verbose > 3)
    line_cluster = get_line_cluster(mask, verbose > 3)
    penalty_cross_center = get_penalty_cross_center(mask)
    goal_bbox = get_goal_bbox(mask)

    key_points_img = get_key_points_mask(rotated_rect_ellipse, line_cluster, penalty_cross_center, goal_bbox)
    if len(key_points_img) < 4:
        return None
    kp_field = get_key_points_top_down_field(FieldDimensions())
    kp_img_list, kp_field_list = get_keypoint_list(key_points_img, kp_field)
    h, status = cv2.findHomography(kp_img_list, kp_field_list)
    robot_pos = get_robot_position_from_homography_matrix(h)[0]

    if verbose > 2:
        class_id_to_color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 122, 0], [0, 122, 122], [122, 0, 122]]
        cmask = util.get_colored_segmentation_mask(mask, class_id_to_color)
        show_keypoint_and_view(get_concept_img(cmask, rotated_rect_ellipse, line_cluster, penalty_cross_center, goal_bbox), key_points_img)

    return robot_pos

def angle_between_2_points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return math.degrees(math.atan2(changeInY,changeInX))

def show_keypoint_and_view(kp_img_drawing, key_points_img):
    kp_field = get_key_points_top_down_field(FieldDimensions())
    kp_img_list, kp_field_list = get_keypoint_list(key_points_img, kp_field)
    h, status = cv2.findHomography(kp_img_list, kp_field_list)
    robot_pos, view_points_in_mask, view_points_in_field_in_mm, look_at_point_in_mm = get_robot_position_from_homography_matrix(h)

    kp_colors = [ran_color(30) for i in range(len(kp_img_list))]
    draw_key_points_on_img(kp_img_drawing, kp_img_list, 9, kp_colors)

    kp_field_drawing, field_drawing_scale = draw_field_key_points(kp_field_list, 500, 3, 10, kp_field, kp_colors)

    view_points_in_field = np.array(view_points_in_field_in_mm * field_drawing_scale)
    robot_pos_in_drawing = [robot_pos.x * field_drawing_scale, robot_pos.y * field_drawing_scale]

    cv2.polylines(kp_img_drawing, np.int32([view_points_in_mask]), 1, (50, 100, 255), 4)
    cv2.polylines(kp_field_drawing, np.int32([view_points_in_field]), 1, (50, 100, 255), 2)
    cv2.circle(kp_field_drawing, (int(robot_pos_in_drawing[0]), int(robot_pos_in_drawing[1])), 10, (50, 100, 255), -1)

    cv2.circle(kp_field_drawing, (int(look_at_point_in_mm[0] * field_drawing_scale), int(look_at_point_in_mm[1]*field_drawing_scale)), 10, (50, 100, 255), -1)

    sight_vector_left = (view_points_in_field[0, :] - view_points_in_field[1, :])*100 + robot_pos_in_drawing
    sight_vector_right = (view_points_in_field[3, :] - view_points_in_field[2, :])*100 + robot_pos_in_drawing

    cv2.line(kp_field_drawing, (int(robot_pos_in_drawing[0]), int(robot_pos_in_drawing[1])), (int(sight_vector_left[0]), int(sight_vector_left[1])), (50, 100, 255), 2)
    cv2.line(kp_field_drawing, (int(robot_pos_in_drawing[0]), int(robot_pos_in_drawing[1])), (int(sight_vector_right[0]), int(sight_vector_right[1])), (50, 100, 255), 2)

    img_array = [
        ["kp_img_drawing", kp_img_drawing],
        ["kp_field_drawing", kp_field_drawing]
    ]
    fig = util.get_figure_of_images(img_array, 2, 1, 10)

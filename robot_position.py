from dataclasses import dataclass

@dataclass
class RobotPosition:
    x: float
    y: float
    rot_left_right_torso: float
    rot_left_right_head: float
    rot_top_down: float
    rot_clockwise: float

def add_robot_postions(pos1, pos2):
    return RobotPosition(
        pos1.x+pos2.x,
        pos1.y+pos2.y,
        pos1.rot_left_right_torso+pos2.rot_left_right_torso,
        pos1.rot_left_right_head+pos2.rot_left_right_head,
        pos1.rot_top_down+pos2.rot_top_down,
        pos1.rot_clockwise+pos2.rot_clockwise
    )

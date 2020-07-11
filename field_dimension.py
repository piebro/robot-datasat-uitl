from dataclasses import dataclass

@dataclass
class FieldDimensions:
    field_x: int=6000
    field_y: int=9000
    line_width: int=50
    penalty_cross_size: int=100
    penalty_area_length: int=600
    penalty_area_width: int=2200
    penalty_cross_distance: int=1300
    center_circle_diameter: int=1500
    border_strip_width: int=700
    goal_width: int=1700

    goal_height: int=800
    goal_post_diameter: int=100

    size_x: int=int(field_x + 2 * border_strip_width)
    size_y: int=int(field_y + 2 * border_strip_width)

    mid_x: int=int(size_x/2)
    mid_y: int=int(size_y/2)

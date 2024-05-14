import cv2
import numpy as np
import sys
sys.path.append('../')
import constants as const
from utils import(convert_meters_to_pixels_util,
                  convert_pixels_to_meters_util,
                  distance,
                  xy_distance,
                  get_bbox_center,
                  get_closest_keypoint_index,
                  get_bbox_height)

# Credit for most of this code goes to abdullahtarek on Github
# A good portion (perhaps 50%) of the code in this file was copied directly from
# his Github project


class Minicourt:
    def __init__(self, frame):
        self.canvas_width = 250
        self.canvas_height = 500
        self.window_corner_offset = 50
        self.minicourt_canvas_padding = 20

        self.set_bgbox_position(frame)
        self.set_minicourt_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixels_util(meters, const.DOUBLE_LINE_WIDTH, self.court_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*(14*2)

        # Manually write-in the (x,y) coords of the keypoints on the minicourt

        # player_feet 0
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        # player_feet 1
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        # player_feet 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_to_pixels(const.HALF_COURT_LINE_HEIGHT*2)
        # player_feet 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_width
        drawing_keypoints[7] = drawing_keypoints[5]
        # #player_feet 4
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_meters_to_pixels(const.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1]
        # #player_feet 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_to_pixels(const.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5]
        # #player_feet 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixels(const.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3]
        # #player_feet 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixels(const.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7]
        # #player_feet 8
        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixels(const.NO_MANS_LAND_HEIGHT)
        # # #player_feet 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixels(const.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17]
        # #player_feet 10
        drawing_keypoints[20] = drawing_keypoints[10]
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixels(const.NO_MANS_LAND_HEIGHT)
        # # #player_feet 11
        drawing_keypoints[22] = drawing_keypoints[20] + self.convert_meters_to_pixels(const.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21]
        # # #player_feet 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17]
        # # #player_feet 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21]

        self.drawing_keypoints = drawing_keypoints

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def set_minicourt_position(self):
        self.court_start_x = self.start_x + self.minicourt_canvas_padding
        self.court_start_y = self.start_y + self.minicourt_canvas_padding
        self.court_end_x = self.end_x - self.minicourt_canvas_padding
        self.court_end_y = self.end_y - self.minicourt_canvas_padding

        self.court_width = self.court_end_x - self.court_start_x
        self.court_height = self.court_end_y - self.court_start_y

    def set_bgbox_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.window_corner_offset
        self.end_y = self.window_corner_offset + self.canvas_height
        self.start_x = self.end_x - self.canvas_width
        self.start_y = self.end_y - self.canvas_height

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

    def draw_bg_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_minicourt(self, frames):
        output_frames = []
        for frame in frames:
            # The function draw_bg_rectangle returns the modified frame,
            frame = self.draw_bg_rectangle(frame)
            # but the function draw_court just draws onto the frame directly
            self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    # The positions parameter is a list of dictionaries (id --> pos) for each frame
    def draw_minicourt_points(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, pos in positions[frame_num].items():
                x, y = pos
                x = int(x)
                y = int(y)
                cv2.circle(frames[frame_num], (x, y), 5, color, cv2.FILLED)
        return frames

    def get_mini_court_coords(self,
                              object_pos,
                              closest_keypoint,
                              closest_keypoint_index,
                              player_height_in_pixels,
                              player_height_in_meters):
        dist_to_key_x_pixels, dist_to_key_y_pixels = xy_distance(closest_keypoint, object_pos)

        dist_to_key_x_meters = convert_pixels_to_meters_util(dist_to_key_x_pixels,
                                                             player_height_in_meters,
                                                             player_height_in_pixels)
        dist_to_key_y_meters = convert_pixels_to_meters_util(dist_to_key_y_pixels,
                                                             player_height_in_meters,
                                                             player_height_in_pixels)

        minicourt_dist_x_pixels = self.convert_meters_to_pixels(dist_to_key_x_meters)
        minicourt_dist_y_pixels = self.convert_meters_to_pixels(dist_to_key_y_meters)
        closest_minicourt_keypoint = (self.drawing_keypoints[2*closest_keypoint_index],
                                       self.drawing_keypoints[2*closest_keypoint_index + 1])

        minicourt_player_position = (closest_minicourt_keypoint[0] + minicourt_dist_x_pixels,
                                     closest_minicourt_keypoint[1] + minicourt_dist_y_pixels)

        return minicourt_player_position

    def convert_bbox_to_pixel_coords(self, player_boxes, ball_boxes, court_keypoints):
        player_heights = {
            1: const.PLAYER_1_HEIGHT_METERS,
            2: const.PLAYER_2_HEIGHT_METERS
        }
        minicourt_player_boxes = []
        minicourt_ball_boxes = []

        for frame_num, player_bboxes in enumerate(player_boxes):
            ball_bbox = ball_boxes[frame_num].get(1)
            ball_pos = get_bbox_center(ball_bbox)
            closest_player_id_to_ball = min(player_bboxes.keys(),
                                            key=lambda x: distance(ball_pos,
                                                                   get_bbox_center(player_bboxes.get(x))
                                                                   )
                                            )

            minicourt_player_boxes_dict = {}
            for player_id, bbox in player_bboxes.items():
                # Measure minimum distance to special reference keypoints
                # and get its index too. Here we use the very top left(id: 0),
                # bottom left(id: 2), and the two middle keypoints(id: 12, 13).

                keypoint_indices = [0, 2, 12, 13]
                # keypoint_indices = [x for x in range(14)]

                player_feet = (int((1/2) * (bbox[0] + bbox[2])), int(bbox[3]))
                closest_keypoint_index = get_closest_keypoint_index(player_feet, court_keypoints, keypoint_indices)
                closest_keypoint = (court_keypoints[2*closest_keypoint_index],
                                    court_keypoints[2*closest_keypoint_index+1])

                # Now return the player height in pixels by taking the maximum bbox heights
                # that has appear in the video around this particular frame
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_bbox_height(player_boxes[i].get(player_id))
                                            for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                # Now get the player's position in pixels on the minicourt
                mini_court_player_pos = self.get_mini_court_coords(player_feet,
                                                                   closest_keypoint,
                                                                   closest_keypoint_index,
                                                                   max_player_height_in_pixels,
                                                                   player_heights[player_id])

                minicourt_player_boxes_dict[player_id] = mini_court_player_pos

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_pos, court_keypoints, keypoint_indices)
                    closest_keypoint = (court_keypoints[2*closest_keypoint_index],
                                        court_keypoints[2*closest_keypoint_index+1])

                    if frame_num % 10 == 0:
                        print("frame_num", frame_num)
                        print("ball_pos", ball_pos)
                        print("closest keypoint id:", closest_keypoint_index)
                        print("closest keypoint coords:", closest_keypoint)

                    mini_court_ball_pos = self.get_mini_court_coords(ball_pos,
                                                                     closest_keypoint,
                                                                     closest_keypoint_index,
                                                                     max_player_height_in_pixels,
                                                                     player_heights[player_id]
                                                                     )

                    minicourt_ball_boxes.append({1: mini_court_ball_pos})
            minicourt_player_boxes.append(minicourt_player_boxes_dict)
        return minicourt_player_boxes, minicourt_ball_boxes

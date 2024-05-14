from utils import (read_video,
                   save_video,
                   distance,
                   draw_player_stats,
                   convert_pixels_to_meters_util
                   )
import constants as const
from trackers import PlayerTracker, BallTracker
from minicourt import Minicourt
from court_line_detector import CourtLineDetector
from copy import deepcopy
import cv2
import pandas as pd


def main():
    # Read Video
    input_video_path = "input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/ball_model_best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections_new.pkl"
                                                 )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect the keypoints
    keypoints_model_path = "models/keypoints_model_learning_rate_e5.pth"
    keypoints_detector = CourtLineDetector(keypoints_model_path)
    court_keypoints = keypoints_detector.predict(video_frames[0])

    # Filter to only the two players playing the game
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print("ball_shot_frames:", ball_shot_frames)

    # Create the Minicourt
    minicourt = Minicourt(video_frames[0])

    # Convert positions to mini court positions
    player_minicourt_detections, ball_minicourt_detections = minicourt.convert_bbox_to_pixel_coords(player_detections,
                                                                                                    ball_detections,
                                                                                                    court_keypoints)
    # Calcualte stats data for each shot
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
        }]

    for ball_shot_frame in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_frame]
        end_frame = ball_shot_frames[ball_shot_frame+1]
        shot_delta_time = (end_frame-start_frame) / 24  # 24fps

        # Get shot distance
        distance_covered_by_ball_pixels = distance(ball_minicourt_detections[start_frame].get(1),
                                                   ball_minicourt_detections[end_frame].get(1))
        distance_covered_by_ball_meters = convert_pixels_to_meters_util(distance_covered_by_ball_pixels,
                                                                        const.DOUBLE_LINE_WIDTH,
                                                                        minicourt.court_width
                                                                        ) 

        # Speed of the ball shot in km/h
        ball_shot_speed = distance_covered_by_ball_meters/shot_delta_time * 2.23694

        # player who the ball
        player_positions = player_minicourt_detections[start_frame]
        player_shot_ball_id = min(player_positions.keys(),
                                  key=lambda player_id:
                                  distance(player_positions.get(player_id),
                                           ball_minicourt_detections[start_frame].get(1)))

        # opponent player speed
        opponent_id = 1 if player_shot_ball_id == 2 else 2
        distance_covered_by_opponent_pixels = distance(player_positions.get(opponent_id),
                                                       player_minicourt_detections[end_frame].get(opponent_id))
        distance_covered_by_opponent_meters = convert_pixels_to_meters_util(distance_covered_by_opponent_pixels,
                                                                            const.DOUBLE_LINE_WIDTH,
                                                                            minicourt.court_width
                                                                            )

        # Calculate speed in mi/hr
        opponent_speed = distance_covered_by_opponent_meters/shot_delta_time * 2.23694

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball_id}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball_id}_total_shot_speed'] += ball_shot_speed
        current_player_stats[f'player_{player_shot_ball_id}_last_shot_speed'] = ball_shot_speed

        current_player_stats[f'player_{opponent_id}_total_player_speed'] += opponent_speed
        current_player_stats[f'player_{opponent_id}_last_player_speed'] = opponent_speed

        player_stats_data.append(current_player_stats)

    # Calculate average speeed of players and shots
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']

    # Draw player and ball detections
    output_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames= ball_tracker.draw_bboxes(output_frames, ball_detections)

    # Draw court keypoints
    output_frames  = keypoints_detector.draw_keypoints_to_video(output_frames, court_keypoints)

    # Draw Minicourt
    output_frames = minicourt.draw_minicourt(output_frames)
    output_frames = minicourt.draw_minicourt_points(output_frames, player_minicourt_detections)
    output_frames = minicourt.draw_minicourt_points(output_frames, ball_minicourt_detections, color=(0,255,255))

    # Draw player stats
    output_frames = draw_player_stats(output_frames, player_stats_data_df)

    # Draw frame number
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()

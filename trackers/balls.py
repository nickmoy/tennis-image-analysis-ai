from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # Extract just the raw positions of the ball
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,
                                         columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5,
                                                                                     min_periods=1,
                                                                                     center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_delta_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_delta_frames_for_hit*1.2)):
            neg_pos_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            plus_pos_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            if neg_pos_change or plus_pos_change:
                change_frame_count = 0
                for j in range(i+1, i + int(minimum_delta_frames_for_hit*1.2) + 1):
                    neg_pos_change_next_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[j] < 0
                    plus_pos_change_next_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[j] > 0

                    if neg_pos_change and neg_pos_change_next_frame:
                        change_frame_count += 1
                    elif plus_pos_change and plus_pos_change_next_frame:
                        change_frame_count += 1

                if change_frame_count > minimum_delta_frames_for_hit-1:
                    # df_ball_positions['ball_hit'].iloc[i] = 1
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frames_with_ball_shots = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frames_with_ball_shots

    # Returns list of Dictionaries (ball_id->bbox), one for each frame
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
                print(f"Ball tracker stubs saved to {stub_path}")

        return ball_detections

    # Returns a dictionary (ball_id --> bbox) with only a single key: id=1
    # since there's only one ball
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        # Dictionary mapping id=1 to ball bbox coordinates
        # in xyxy format
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, frames, ball_detections):
        output_frames = []
        color = (0, 255, 255)
        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            output_frames.append(frame)

        return output_frames

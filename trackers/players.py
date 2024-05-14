from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import get_bbox_center, distance


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, keypoints, player_detections):
        # Filter only the two tennis players
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(keypoints,
                                             player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id,
                                    bbox in player_dict.items()
                                    if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, keypoints, player_dict):
        distances = []
        for player_id, bbox in player_dict.items():
            player_center = get_bbox_center(bbox)
            min_dist = float('inf')
            for i in range(len(keypoints), 2):
                keypoint = (keypoints[i], keypoints[i+1])
                dist = distance(player_center, keypoint)
                if dist < min_dist:
                    min_dist = dist
            distances.append((player_id, min_dist))
        distances.sort(key=lambda x: x[1])
        # Return ID's of the chosen players
        return [distances[0][0], distances[1][0]]

    # Returns list of Dictionaries (player id->bbox), one for each frame
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
                print(f"Player tracker stubs saved to {stub_path}")

        return player_detections

    # Returns a dictionary (player_id --> bbox) of all player id's and bboxes
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, conf=0.2)[0]
        id_name_dict = results.names

        # Dictionary mapping tracker id to player bounding box coordinates
        # in xyxy format
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames

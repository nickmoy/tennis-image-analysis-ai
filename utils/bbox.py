import numpy as np
# bboxes are arrays of the format (x1, y1, x2, y2)

# Returns center of bbox
def get_bbox_center(bbox):
    return (int((1/2)*(bbox[0] + bbox[1])), int((1/2)*(bbox[2] + bbox[3])))


def get_bbox_height(bbox):
    return (bbox[3] - bbox[1])


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def xy_distance(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def get_closest_keypoint_index(point, court_keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = (court_keypoints[2*keypoint_index], court_keypoints[2*keypoint_index+1])
        distance = abs(point[1]-keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind

import numpy as np
import scipy.io

ACTIVITY = ['CARDS', 'CHESS', 'JENGA', 'PUZZLE']
LOCATION = ['COURTYARDS', 'OFFICE', 'LIVINGROOM']
VIEWER = ['B', 'H', 'S', 'T']
PARTNER = ['B', 'H', 'S', 'T']


def segmentation2box(shape):
    # Find min/max of x and y coordinates
    box_xyxy = np.round([np.min(shape[:, 0]), np.min(shape[:, 1]), np.max(shape[:, 0]), np.max(shape[:, 1])]).astype(int)
    
    # Ensure the box coordinates are within the image boundaries (assuming 1280x720 resolution)
    box_xyxy[0] = max(1, box_xyxy[0])
    box_xyxy[1] = max(1, box_xyxy[1])
    box_xyxy[2] = min(1280, box_xyxy[2])
    box_xyxy[3] = min(720, box_xyxy[3])
    
    # Convert from xyxy (x_min, y_min, x_max, y_max) to xywh (x_min, y_min, width, height)
    box_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]]
    
    return box_xywh

def get_data_from_struct(vid):
    result = []
    for video in vid[0]: 
        video_name = ''
        if 'activity_id' in video.dtype.names and len(video['activity_id']) != 0:
            video_name += video['activity_id'][0] + '_'
        if 'location_id' in video.dtype.names and len(video['location_id']) != 0:
            video_name += video['location_id'][0] + '_'
        if 'ego_viewer_id' in video.dtype.names and len(video['ego_viewer_id']) != 0:
            video_name += video['ego_viewer_id'][0] + '_'
        if 'partner_id' in video.dtype.names and len(video['partner_id']) != 0:
            video_name += video['partner_id'][0]

        for frame in video['labelled_frames'][0]:
            dictionary = {}
            boxes = np.zeros((4, 4))  # Initialize an array for 4 bounding boxes (each with 4 values)
            if 'myleft' in frame.dtype.names and len(frame['myleft']) != 0:
                boxes[0, :] = segmentation2box(np.array(frame['myleft']))
            if 'myright' in frame.dtype.names and len(frame['myright']) != 0:
                boxes[1, :] = segmentation2box(np.array(frame['myright']))
            if 'yourleft' in frame.dtype.names and len(frame['yourleft']) != 0:
                boxes[2, :] = segmentation2box(np.array(frame['yourleft']))
            if 'yourright' in frame.dtype.names and len(frame['yourright']) != 0:
                boxes[3, :] = segmentation2box(np.array(frame['yourright']))
            if 'frame_num' in frame.dtype.names and len(frame['frame_num']) != 0:
                frame_nums = frame['frame_num'][0].item()

            dictionary = {
                'boxes': boxes,
                'video_name': video_name,
                'frame_num': frame_nums,
            }
            result.append(dictionary)

    return result


def get_annotations(file_path):
    mat = scipy.io.loadmat(file_path)
    struct = mat['video']

    return get_data_from_struct(struct)


# annotations = get_annotations('EgoHand_dataset/metadata.mat')
# print(annotations[0])
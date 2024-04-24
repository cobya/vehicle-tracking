# Forked from https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation at commit 7c2aad194a146de668ee5d9981baa9fbb12f3b52
import argparse
from collections import defaultdict, deque
import csv

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

SOURCE = np.array([[150, 550], [775, 400], [1400, 425], [900, 800]])

# Meters
TARGET_WIDTH = 15
TARGET_HEIGHT = 15

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

DETECTIONS_CSV_DATA = {}

class DetectionRowEntry():
    def __init__(self):
        self.TrackerId = None
        self.FirstFrame = None
        self.FirstTimeSeconds = None
        self.MaxSpeed = None
        self.VehicleType = None

def ComputeLabelFromClassId(classId):
    """
    Function to add vehicle type label by class
    """
    if classId == 0: # Person
        return "Person"
    elif classId == 2: # Car
        return "Car"
    elif classId == 3: # Motobike
        return "Motorcycle"
    elif classId == 5: # Bus
        return "Bus"
    elif classId == 7: # Truck
        return "Truck"
    else:
        return "Unknown"

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--target_csv_path",
        required=False,
        default="traffic.csv",
        help="Path to the target csv file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--show",
        required=False,
        default=True,
        help="Whether or not to display the image being calculated.",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # initialize .csv
    with open(args.target_csv_path, 'w') as f:
        writer = csv.writer(f)
        csv_line = \
            'Frame, Seconds, Detection ID, Vehicle Type, Vehicle Speed (MPH)'
        writer.writerows([csv_line.split(',')])

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=args.confidence_threshold
    )

    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)

    thickness = round(thickness / 2.0)
    text_scale = round(text_scale / 2.0)

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    frameCount = 0

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            if frameCount % 100 == 0:
                print(f"-----\t----------------------------\t-----")
                print(f"-----\tFRAME ITERATION {frameCount} OF {video_info.total_frames}\t-----")
                print(f"-----\t----------------------------\t-----")

            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for _, _, _, class_id, tracker_id, _ in detections:
                # generate CSV data
                if tracker_id not in DETECTIONS_CSV_DATA and class_id in [0, 2, 3, 5, 7]: # filter to vehicles and persons
                    DETECTIONS_CSV_DATA[tracker_id] = DetectionRowEntry()
                    DETECTIONS_CSV_DATA[tracker_id].TrackerId = tracker_id
                    DETECTIONS_CSV_DATA[tracker_id].FirstFrame = frameCount
                    DETECTIONS_CSV_DATA[tracker_id].FirstTimeSeconds = frameCount / video_info.fps
                    DETECTIONS_CSV_DATA[tracker_id].MaxSpeed = 0
                    DETECTIONS_CSV_DATA[tracker_id].VehicleType = ComputeLabelFromClassId(class_id)

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = float(abs(coordinate_start - coordinate_end))
                    time = len(coordinates[tracker_id]) / float(video_info.fps)
                    speed = distance / time * 3.6
                    speed *= 0.621 # convert to MPH
                    labels.append(f"#{tracker_id} | {ComputeLabelFromClassId(class_id)} | {int(speed)} MPH")

                    # update any existing CSV entries
                    if tracker_id in DETECTIONS_CSV_DATA:
                        DETECTIONS_CSV_DATA[tracker_id].MaxSpeed = max(speed, DETECTIONS_CSV_DATA[tracker_id].MaxSpeed)

            annotated_frame = frame.copy()
            annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(annotated_frame)

            if args.show is True:
                cv2.imshow("frame", annotated_frame)
            frameCount += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # CSV generation
        # Frame, Seconds, Detection ID, Vehicle Type, Vehicle Speed (MPH)
        with open(args.target_csv_path, 'a') as f:
            writer = csv.writer(f)
            for tracker_id in DETECTIONS_CSV_DATA:
                detection = DETECTIONS_CSV_DATA[tracker_id]
                # filter out stationary only vehicles
                if (detection.MaxSpeed > 3 and detection.VehicleType == "Car") or (detection.MaxSpeed > 1 and detection.VehicleType != "Car"):
                    csv_line_append = f"#{detection.FirstFrame},{detection.FirstTimeSeconds},{detection.TrackerId},{detection.VehicleType},{detection.MaxSpeed}"
                    writer.writerows([csv_line_append.split(',')])

        # exit logic
        cv2.destroyAllWindows()
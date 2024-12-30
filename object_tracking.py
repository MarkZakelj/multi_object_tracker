"""
Track multiple objects in a video using deep sort algorithm
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from database import DBInterface
from mobilenet import MobileNetEmbedder

DISPLAY = False

class ObjectTracker:
    def __init__(self, yolo_weights='yolov4.weights', yolo_cfg='yolov4.cfg', target_class='car', db_name='tracking.db'):
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)
        self.embedder = MobileNetEmbedder()
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open('data/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.target_class_id = self.classes.index(target_class)
        self.tracker = DeepSort(max_age=20, nms_max_overlap=0.85, embedder=None)
        self.db_name = db_name
        
    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        bbs = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == self.target_class_id and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    bbs.append(([x, y, w, h], float(confidence), class_id))
        
        return bbs
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        db = DBInterface(db_path=self.db_name)
        
        while True:
            ret, frame = cap.read()
        
            if not ret:
                break
            
            frame_count += 1
            bbs = self.detect_objects(frame)
            embeds = []
            for bb in bbs:
                x, y, w, h = bb[0]
                embed = self.embedder.get_embedding(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
                embeds.append(embed)
            embeds = np.array(embeds, dtype=np.float32)
            
            
            if len(bbs) > 0:
                tracks = self.tracker.update_tracks(bbs, frame=frame, embeds=embeds)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    conf = track.det_conf
                    if conf is None:
                        conf = 0.0
                    class_name = self.classes[int(track.det_class)]
                    print("Tracker ID: {}, Conf: {:.3f}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), conf, class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    db.add_detection(object_id=2, object_name='car', track_id=track_id, 
                                     confidence=conf, video_id=video_path, frame=frame_count,
                                     bbox=bbox)
                    
                    if DISPLAY:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id} - {conf:.3f} conf", (int(bbox[0]), int(bbox[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Processed frame {frame_count}")
            print()
            
            if DISPLAY:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nProcessing complete. Results saved to database")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/cars.mp4', help='Path to input video file')
    parser.add_argument('--weights', type=str, default='model/yolov4.weights', help='Path to YOLO weights')
    parser.add_argument('--cfg', type=str, default='model_cfg/yolov4.cfg', help='Path to YOLO config')
    
    args = parser.parse_args()
    
    tracker = ObjectTracker(args.weights, args.cfg, target_class='car', db_name='tracking2.db')
    tracker.process_video(args.video)

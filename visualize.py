import cv2
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from database import get_track_ids, get_trajectory_with_confidence, get_videos

def get_video_info(video_path):
   cap = cv2.VideoCapture(video_path)
   frames = []
   
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   dimensions = (width, height)
   
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
           
   cap.release()
   return frames, dimensions


def color_interpolation(frame, max_frame, color1=(0,0,255), color2=(255,0,0)):
    ratio = frame / max_frame
    return tuple(int(c1 + (c2-c1)*ratio) for c1, c2 in zip(color1, color2))

def create_visualization(data, show_boxes=True, single_frame=None, dimensions=None, actual_frame=None):
    if not data:
        return None
    if dimensions is None:
        max_x = max(max(r for _, l, t, r, b, _ in data), 1920)
        max_y = max(max(b for _, l, t, r, b, _ in data), 1080)
    else:
        max_x = dimensions[0]
        max_y = dimensions[1]
    
    if actual_frame is None:
        img = Image.new('RGB', (int(max_x), int(max_y)), (235, 235, 235))
    else:
        img = Image.fromarray(actual_frame)
    draw = ImageDraw.Draw(img)
    
    frames = [row[0] for row in data]
    max_frame = max(frames)
    
    font = ImageFont.load_default(size=40)

    if single_frame is not None:
        frame_data = [d for d in data if d[0] == single_frame]
        if frame_data:
            frame, l, t, r, b, conf = frame_data[0]
            color = color_interpolation(frame, max_frame)
            
            if l < r and t < b:
                draw.rectangle([l, t, r, b], outline=color, width=4)
            
            conf_text = f"Conf: {conf:.2f}"
            fill = 'black' if actual_frame is None else 'white'
            draw.text((l, t-40), conf_text, fill=fill, font=font, stroke_width=1)
            
            center_x = (l + r) / 2
            center_y = (t + b) / 2
            dot_size = 6
            draw.ellipse([center_x-dot_size, center_y-dot_size, 
                         center_x+dot_size, center_y+dot_size], 
                         fill=color, width=4)
    else:
        for frame, l, t, r, b, conf in data:
            color = color_interpolation(frame, max_frame)
            center_x = (l + r) / 2
            center_y = (t + b) / 2
            
            dot_size = 4
            draw.ellipse([center_x-dot_size, center_y-dot_size, 
                         center_x+dot_size, center_y+dot_size], 
                         fill=color)
            
            if show_boxes:
                if l < r and t < b:
                    draw.rectangle([l, t, r, b], outline=color, width=2)
    
    return img

def main():
    st.title("Object Tracking Visualization")
    
    with st.sidebar:
        videos = get_videos()
        selected_video = st.selectbox("Select Video", videos)
        track_ids = get_track_ids(selected_video)
        selected_track = st.selectbox("Select Track ID", track_ids)
        single_frame_view = st.checkbox("Single Frame View", False)
        show_boxes = st.checkbox("Show Bounding Boxes", False)
        st.text("Applicable to single-frame view:")
        show_actual_image = st.checkbox("Show actual image", False)
        
    image_frames, dimensions = get_video_info('data/cars.mp4')
    
    if selected_track:
        data = get_trajectory_with_confidence(selected_track)
        
        if data:
            frames = [row[0] for row in data]
            max_frame = max(frames)
            min_frame = min(frames)
            
            if single_frame_view:
                frame = st.slider("Frame", min_frame, max_frame, min_frame)
                actual_frame = None
                if show_actual_image:
                    actual_frame = image_frames[frame-1]
                img = create_visualization(data, show_boxes, single_frame=frame, 
                                           dimensions=dimensions, actual_frame=actual_frame)
            else:
                img = create_visualization(data, show_boxes, dimensions=dimensions)
            
            if img:
                st.image(img, caption=f"Track ID: {selected_track}")
            if not single_frame_view:
                dframe = pd.DataFrame(data=data, columns=['Frame', 'Left', 'Top', 'Right', 'Bottom', 'Confidence'])
                for col in ['Left', 'Top', 'Right', 'Bottom']:
                    dframe[col] = dframe[col].astype(int)
                st.header("Info Table")
                st.dataframe(dframe)
                


if __name__ == "__main__":
    main()
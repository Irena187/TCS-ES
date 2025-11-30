import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import json 
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
from gpiozero import LED

# --- Define the file to write status to ---
STATUS_FILE = "/home/UKIRMA/hailo-rpi5-examples/basic_pipelines/traffic_status.json"

def update_status_file(preferred_street, count_a, count_b):
    """
    Helper function to write the current status to a JSON file.
    'preferred_street' should be 'A' or 'B'.
    """
    status = {
        "preferred_street": preferred_street,
        "street_a_count": count_a,
        "street_b_count": count_b,
    }
    try:
        # --- FIX: Write to the project root, not the script's directory ---
        # This ensures traffic_status.json is in the same folder as index.html
        status_file_path = os.path.join(project_root, STATUS_FILE)
        with open(status_file_path, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Error writing status file: {e}")

# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        
        # --- 1. Configuration: Objects to detect ---
        self.target_vehicles = ["car", "bus", "truck"]
        self.confidence_threshold = 0.4

        # --- 2. Configuration: Zone Coordinates (0.0 - 1.0) ---
        # !! YOU MUST ADJUST THESE to match your camera feed !!
        # Street A (Vertical Road)
        self.zone1 = {'xmin': 0.0, 'xmax': 0.5, 'ymin': 0.5, 'ymax': 1.0} # Top section
        self.zone2 = {'xmin': 0.5, 'xmax': 1.0, 'ymin': 0.0, 'ymax': 0.5} # Bottom section
        
        # Street B (Horizontal Road)
        self.zone3 = {'xmin': 0.5, 'xmax': 1.0, 'ymin': 0.5, 'ymax': 1.0} # Left section
        self.zone4 = {'xmin': 0.0, 'xmax': 0.5, 'ymin': 0.0, 'ymax': 0.5} # Right section

        # --- 3. Configuration: GPIO Pins ---
        # !! YOU MUST ADJUST THESE to match your RPi wiring !!
        # Street A LEDs
        self.led_A_green = LED(17) 
        self.led_A_red = LED(18)
        # Street B LEDs
        self.led_B_green = LED(22)
        self.led_B_red = LED(23)

        # --- 4. State & Debouncing Variables ---
        self.street_A_preferred = True  # Start with Street A as Green
        self.prefer_A_frames = 0        # Consecutive frames preferring A
        self.prefer_B_frames = 0        # Consecutive frames preferring B
        self.debounce_frames_to_change = 5 # Frames to wait before switching

        # --- 5. Initial LED State & Status File ---
        print("Initializing lights: Street A = GREEN, Street B = RED")
        self.led_A_green.on()
        self.led_A_red.off()
        self.led_B_green.off()
        self.led_B_red.on()
        
        # --- Write initial state to file ---
        update_status_file("A", 0, 0)

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # --- Initialize counts for this frame ---
    count_sec1, count_sec2, count_sec3, count_sec4 = 0, 0, 0, 0
    
    # --- Parse the detections ---
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        
        if confidence > user_data.confidence_threshold and label in user_data.target_vehicles:
            bbox = detection.get_bbox()
            center_x = bbox.xmin() + (bbox.width() / 2)
            center_y = bbox.ymin() + (bbox.height() / 2)
            
            # --- Check which zone the object is in ---
            z1 = user_data.zone1
            if (z1['xmin'] <= center_x <= z1['xmax'] and z1['ymin'] <= center_y <= z1['ymax']):
                count_sec1 += 1
                
            z2 = user_data.zone2
            if (z2['xmin'] <= center_x <= z2['xmax'] and z2['ymin'] <= center_y <= z2['ymax']):
                count_sec2 += 1
                
            z3 = user_data.zone3
            if (z3['xmin'] <= center_x <= z3['xmax'] and z3['ymin'] <= center_y <= z3['ymax']):
                count_sec3 += 1
                
            z4 = user_data.zone4
            if (z4['xmin'] <= center_x <= z4['xmax'] and z4['ymin'] <= center_y <= z4['ymax']):
                count_sec4 += 1

    # --- Calculate Total Counts for each street ---
    count_street_A = count_sec1 + count_sec2
    count_street_B = count_sec3 + count_sec4
    
    print(f"Frame Counts: Street A = {count_street_A} | Street B = {count_street_B}")

    # --- Debouncing & Decision Logic ---
    if count_street_A < count_street_B:
        user_data.prefer_A_frames += 1
        user_data.prefer_B_frames = 0
    elif count_street_B < count_street_A:
        user_data.prefer_B_frames += 1
        user_data.prefer_A_frames = 0
    else: 
        user_data.prefer_A_frames = 0
        user_data.prefer_B_frames = 0

    # --- State Change Logic ---
    
    # Check if we need to switch TO Street A
    if (user_data.prefer_A_frames >= user_data.debounce_frames_to_change and 
        not user_data.street_A_preferred):
        
        print("--- SWITCHING: Street A GREEN, Street B RED ---")
        user_data.street_A_preferred = True
        
        user_data.led_A_green.on()
        user_data.led_A_red.off()
        user_data.led_B_green.off()
        user_data.led_B_red.on()
        
        # --- Update status file on change ---
        update_status_file("A", count_street_A, count_street_B)
        user_data.prefer_A_frames = 0

    # Check if we need to switch TO Street B
    elif (user_data.prefer_B_frames >= user_data.debounce_frames_to_change and 
          user_data.street_A_preferred):
        
        print("--- SWITCHING: Street B GREEN, Street A RED ---")
        user_data.street_A_preferred = False
        
        user_data.led_A_green.off()
        user_data.led_A_red.on()
        user_data.led_B_green.on()
        user_data.led_B_red.off()
        
        # --- Update status file on change ---
        update_status_file("B", count_street_A, count_street_B)
        user_data.prefer_B_frames = 0
        
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
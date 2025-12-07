import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
from inference_sdk import InferenceHTTPClient
import os
import json
import cv2
import numpy as np
import threading
import time

class HeadcountApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Headcount Detection")
        self.root.geometry("800x700")
        
        # Initialize API client
        self.CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="Y3H27dF2JeXEOpwxBCof"  # keep this safe; don't share publicly
        )
        
        self.image_path = None
        self.photo = None
        self.result = None
        self.camera_active = False
        self.cap = None
        self.camera_thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Select Image button
        self.select_btn = ttk.Button(
            button_frame, 
            text="Select Image", 
            command=self.select_image
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Scan button (initially disabled)
        self.scan_btn = ttk.Button(
            button_frame,
            text="Scan Image",
            command=self.scan_image,
            state=tk.DISABLED
        )
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera button
        self.camera_btn = ttk.Button(
            button_frame,
            text="Use Camera",
            command=self.toggle_camera
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_label = ttk.Label(main_frame, background='#f0f0f0')
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results display
        self.result_label = ttk.Label(
            main_frame, 
            text="No image selected",
            wraplength=780,
            justify=tk.CENTER
        )
        self.result_label.pack(fill=tk.X, pady=10)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image()
            self.scan_btn.config(state=tk.NORMAL)
            self.result_label.config(text="Click 'Scan Image' to process the selected image")
    
    def display_image(self):
        try:
            # Open and resize the image
            image = Image.open(self.image_path)
            # Resize image to fit in the window while maintaining aspect ratio
            max_size = (780, 500)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        self.camera_active = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.camera_active = False
            return
            
        self.camera_btn.config(text="Stop Camera")
        self.select_btn.config(state=tk.DISABLED)
        self.scan_btn.config(state=tk.DISABLED)
        
        # Start camera in a separate thread
        self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
        self.camera_thread.start()
    
    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.camera_btn.config(text="Use Camera")
        self.select_btn.config(state=tk.NORMAL)
        self.scan_btn.config(state=tk.NORMAL if self.image_path else tk.DISABLED)
        
        # Clear the image display
        self.image_label.config(image='')
        self.image_label.image = None
    
    def update_camera(self):
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert the frame to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform inference
            result = self.CLIENT.infer(frame_rgb, model_id="head-detection/2")
            
            # Draw bounding boxes
            if 'predictions' in result:
                for pred in result['predictions']:
                    x = int(pred['x'] - pred['width']/2)
                    y = int(pred['y'] - pred['height']/2)
                    w = int(pred['width'])
                    h = int(pred['height'])
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert to PIL Image and display
            image = Image.fromarray(frame_rgb)
            image.thumbnail((780, 500), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            
            # Update the UI in the main thread
            self.root.after(0, self.update_display, self.photo, len(result.get('predictions', [])))
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_display(self, photo, head_count):
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.result_label.config(text=f"Detected {head_count} heads")
    
    def scan_image(self):
        if not self.image_path or not os.path.exists(self.image_path):
            messagebox.showerror("Error", "No valid image selected")
            return
        
        self.scan_btn.config(state=tk.DISABLED, text="Scanning...")
        self.result_label.config(text="Processing image, please wait...")
        self.root.update()  # Update the UI
        
        try:
            # Run inference
            self.result = self.CLIENT.infer(self.image_path, model_id="crowd-counting-dataset-w3o7w/2")
            
            # Save results
            with open("inference_result.json", "w") as f:
                json.dump(self.result, f, indent=4)
            
            # Display results
            predictions = self.result.get("predictions", [])
            count = len(predictions)
            self.result_label.config(
                text=f"✅ Scan complete!\n\n"
                     f"🧍 Detected {count} {'person' if count == 1 else 'people'} in the image.\n"
                     f"📄 Results saved to: inference_result.json"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.result_label.config(text="Error processing image. Please try again.")
        
        self.scan_btn.config(state=tk.NORMAL, text="Scan Image")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeadcountApp(root)
    root.mainloop()

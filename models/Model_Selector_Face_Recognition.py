import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the model architecture (same as in your training script)
def build_model(num_classes):
    # Load pre-trained VGG16 model
    model = models.vgg16(weights=None)  # No need to download weights again

    # Modify classifier for our number of classes
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, num_classes)
    )

    return model


# Function to predict a person's identity with face detection
def predict_person_with_face_detection(model, image_path, label_dict):
    # Load the image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load a pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if any faces were detected
    if len(faces) == 0:
        print("No face detected in the image. Processing the entire image instead.")
        # Process the entire image as fallback
        image = cv2.resize(original_image, (224, 224))
        display_img = original_image.copy()
    else:
        # Process the largest face (assuming it's the main subject)
        largest_face = None
        largest_area = 0

        for (x, y, w, h) in faces:
            if w * h > largest_area:
                largest_area = w * h
                largest_face = (x, y, w, h)

        # Extract the face region
        x, y, w, h = largest_face
        image = original_image[y:y + h, x:x + w]

        # Draw rectangle around the detected face
        display_img = original_image.copy()
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Resize to 224x224 for VGG16
        image = cv2.resize(image, (224, 224))

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][preds[0]].item()

    # Get the person's name
    predicted_class = preds[0].item()
    person_name = label_dict[predicted_class]

    return person_name, confidence, display_img, image


# Function to scan for available models and label dictionaries
def scan_models_and_dictionaries():
    models = []
    dicts = []

    # Look for model files (*.pth)
    for file in os.listdir('.'):
        if file.endswith('.pth'):
            models.append(file)

    # Look for dictionary files (*.pkl)
    for file in os.listdir('.'):
        if file.endswith('.pkl'):
            dicts.append(file)

    return models, dicts


# Main application class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Model Selector")
        self.root.geometry("700x500")

        # Create frame for model selection
        frame = ttk.LabelFrame(root, text="Model Selection")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Scan for models and dictionaries
        self.model_files, self.dict_files = scan_models_and_dictionaries()

        # Model selector
        ttk.Label(frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(frame, textvariable=self.model_var, width=50)
        self.model_dropdown['values'] = self.model_files
        if self.model_files:
            # Try to select a finetuned model first if available
            finetuned_models = [m for m in self.model_files if 'finetuned' in m.lower()]
            if finetuned_models:
                self.model_dropdown.set(finetuned_models[0])
            else:
                self.model_dropdown.set(self.model_files[0])
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Dictionary selector
        ttk.Label(frame, text="Select Label Dictionary:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dict_var = tk.StringVar()
        self.dict_dropdown = ttk.Combobox(frame, textvariable=self.dict_var, width=50)
        self.dict_dropdown['values'] = self.dict_files
        if self.dict_files:
            # Try to select matching dictionary if available
            if 'filtered' in self.model_var.get().lower() and any('filtered' in d.lower() for d in self.dict_files):
                filtered_dicts = [d for d in self.dict_files if 'filtered' in d.lower()]
                self.dict_dropdown.set(filtered_dicts[0])
            elif any('original' in d.lower() for d in self.dict_files):
                original_dicts = [d for d in self.dict_files if 'original' in d.lower()]
                self.dict_dropdown.set(original_dicts[0])
            else:
                self.dict_dropdown.set(self.dict_files[0])
        self.dict_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Autodetect appropriate dictionary
        ttk.Button(frame, text="Auto-Match Dictionary", command=self.auto_match_dict).grid(row=2, column=0,
                                                                                           columnspan=2, padx=5, pady=5)

        # Model info display
        self.info_text = tk.Text(frame, height=6, width=80)
        self.info_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        self.update_model_info()

        # Bind events to update info
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model_info)
        self.dict_dropdown.bind("<<ComboboxSelected>>", self.update_model_info)

        # Run recognition button
        ttk.Button(frame, text="Select Image and Run Recognition", command=self.run_recognition).grid(row=4, column=0,
                                                                                                      columnspan=2,
                                                                                                      padx=5, pady=20)

    def auto_match_dict(self):
        model_name = self.model_var.get()

        if not model_name or not self.dict_files:
            return

        # Logic to match dictionary with model
        if 'filtered' in model_name.lower():
            filtered_dicts = [d for d in self.dict_files if 'filtered' in d.lower()]
            if filtered_dicts:
                self.dict_dropdown.set(filtered_dicts[0])
        elif any(str(c) in model_name for c in range(10)):
            # If model name contains a number (like classes count)
            for n in range(1, 1000):
                if f"{n}_classes" in model_name:
                    matching_dicts = [d for d in self.dict_files if f"{n}" in d]
                    if matching_dicts:
                        self.dict_dropdown.set(matching_dicts[0])
                    break
        else:
            # Default to original model
            original_dicts = [d for d in self.dict_files if 'original' in d.lower() or 'label_dict.pkl' == d]
            if original_dicts:
                self.dict_dropdown.set(original_dicts[0])

        self.update_model_info()

    def update_model_info(self, event=None):
        model_name = self.model_var.get()
        dict_name = self.dict_var.get()

        self.info_text.delete(1.0, tk.END)

        if not model_name or not dict_name:
            self.info_text.insert(tk.END, "Please select both a model and a label dictionary.")
            return

        # Try to load the dictionary to get class count
        try:
            with open(dict_name, 'rb') as f:
                label_dict = pickle.load(f)
                num_classes = len(label_dict)
        except Exception as e:
            num_classes = "Unknown"

        # Extract info from model name
        model_type = "Fine-tuned" if "finetuned" in model_name.lower() else "Base"

        info = f"Selected Model: {model_name}\n"
        info += f"Selected Dictionary: {dict_name}\n"
        info += f"Model Type: {model_type}\n"
        info += f"Number of Classes: {num_classes}\n\n"

        # Validation check
        if "filtered" in model_name.lower() and "filtered" not in dict_name.lower():
            info += "WARNING: You selected a filtered model but not a filtered dictionary!\n"
        elif "filtered" not in model_name.lower() and "filtered" in dict_name.lower():
            info += "WARNING: You selected a filtered dictionary but not a filtered model!\n"

        self.info_text.insert(tk.END, info)

    def run_recognition(self):
        model_name = self.model_var.get()
        dict_name = self.dict_var.get()

        if not model_name or not dict_name:
            tk.messagebox.showwarning("Selection Error", "Please select both a model and a label dictionary.")
            return

        # Load the label dictionary
        try:
            with open(dict_name, 'rb') as f:
                label_dict = pickle.load(f)
            print(f"Loaded label dictionary with {len(label_dict)} classes")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load dictionary: {e}")
            return

        # Load the model
        try:
            num_classes = len(label_dict)
            model = build_model(num_classes)
            model.load_state_dict(torch.load(model_name, map_location=device))
            model = model.to(device)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # File dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*"))
        )

        if not file_path:
            print("No file selected")
            return

        # Run prediction
        try:
            person_name, confidence, display_img, face_img = predict_person_with_face_detection(model, file_path,
                                                                                                label_dict)

            print(f"Predicted person: {person_name}")
            print(f"Confidence: {confidence:.2%}")

            # Create a new window for the plot instead of using plt.show() directly
            plot_window = tk.Toplevel(self.root)
            plot_window.title(f"Prediction: {person_name}")
            plot_window.geometry("1000x500")

            # Use matplotlib with TkAgg backend explicitly
            import matplotlib
            matplotlib.use('TkAgg')
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Create figure and axes
            fig = Figure(figsize=(12, 6))

            # Original image with face detection box
            ax1 = fig.add_subplot(121)
            ax1.imshow(display_img)
            ax1.set_title("Original Image")
            ax1.axis('off')

            # Extracted face used for recognition
            ax2 = fig.add_subplot(122)
            ax2.imshow(face_img)
            ax2.set_title(f"Predicted: {person_name} ({confidence:.2%})")
            ax2.axis('off')

            fig.tight_layout()

            # Place the plot on the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Error during prediction: {e}")
            print(f"Error details: {e}")


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
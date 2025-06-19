import cv2
from model import AccidentDetectionModel
import numpy as np
import os

# Load the trained model
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the image path from a different folder
image_path = r"C:\Users\admin\OneDrive\Desktop\Accident-Detection-System-main\data\test\Accident\test30_8.jpg"    # Modify this with the correct path

def process_image(image_path):
    # Load the image from the specified path
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Unable to open image.")
        return

    # Process the image (resize and predict)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)

        # Optional alert (Mac only):
        # if prob > 90:
        #     os.system("say beep")

        cv2.rectangle(image, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(image, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

    # Display the processed image
    cv2.imshow('Accident Detection', image)

    # Wait for key press to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_image(image_path)

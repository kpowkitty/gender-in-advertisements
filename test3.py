import os
import pandas as pd
from deepface import DeepFace

Specify the folder containing your images
image_folder = "~/Pictures/ads90s"

Create a list to store the results
results = []

Loop through all files in the folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    # Ensure the file is an image (you can add more checks based on your file types)
    if image_path.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Analyzing image: {filename}")

        # Analyze the image for gender
        result = DeepFace.analyze(image_path, actions=['gender'])

        # Append the filename and gender prediction to the results list
        results.append({'Image': filename, 'Predicted Gender': result[0]['gender']})

Create a Pandas DataFrame from the results
df = pd.DataFrame(results)

Save the DataFrame to an Excel file
df.to_excel("image_predictions.xlsx", index=False)

print("Predictions saved to image_predictions.xlsx")

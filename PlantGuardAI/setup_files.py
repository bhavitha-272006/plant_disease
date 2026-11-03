import os, json
import pandas as pd
from PIL import Image, ImageDraw

for d in ["assets", "data/sample_leaves", "models"]:
    os.makedirs(d, exist_ok=True)

# assets/logo.png (green leaf icon)
img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
dr = ImageDraw.Draw(img)
dr.ellipse((40, 60, 216, 200), fill=(25, 230, 176, 255))
dr.line((60, 190, 200, 70), fill=(6, 32, 51, 255), width=6)
dr.line((128, 200, 128, 240), fill=(25, 230, 176, 255), width=10)
img.save("assets/logo.png")

# data/history.csv
pd.DataFrame(columns=["Time", "Plant", "Result", "Confidence"]).to_csv(
    "data/history.csv", index=False)

# models/class_names.json (placeholder, replaced by the Colab file later)
classes = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
json.dump(classes, open("models/class_names.json", "w"), indent=2)

print("Done. Files created in assets, data, models.")
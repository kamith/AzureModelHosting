from contextlib import asynccontextmanager
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw
from base64 import b64decode
import io
import uvicorn

class ImageModel(BaseModel): 
    data: str

# Variables
MODEL_PATH = "model.pt"

# Context dictionary
context = {}

# Expected height and width
EXPECTED_HEIGHT = 640
EXPECTED_WIDTH = 640


@asynccontextmanager
async def lifespan(app: FastAPI):
    context["model"] = YOLO(MODEL_PATH)
    yield
    # Free resources
    context.clear()
    print("Resources cleared")

# Start server
app = FastAPI(lifespan=lifespan)

# Convert image string to file
def convert_to_file(data):
    split = data.split(',')
    imageStr = split[1] if len(split) > 1 else data
    image = Image.open(io.BytesIO(b64decode(imageStr)))
    image = image.resize((EXPECTED_WIDTH, EXPECTED_HEIGHT))
    return np.array(image)

@app.post("/detect")
async def detect(str: str = Body(...)): 
    model = context["model"]
    
    # Convert base64 to image
    image_array = convert_to_file(str)
    
    # Get prediction from model
    try:
        results = model.predict(
            source=image_array,  # Use numpy array instead of file path
            conf=0.4,  # Confidence threshold
        )
    except:
        raise HTTPException(500)
    
    # Get result
    result = results[0]
    
    # Get class names
    class_names = model.names

    violations = []
    detections = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        class_name = class_names[class_id]
        confidence = float(box.conf.item())
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
        "class": class_name,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2]
        })
    
    # Check if obstruction class was detected
    obstruction_detected = any(det["class"] == "Obstruction" for det in detections)
    washing_station_detected = any(det["class"] == "Washing Station Area" for det in detections)
    washing_station_detected = any(det["class"] == "Washing Station" for det in detections)
    fire_extinguisher_detected = any(det["class"] == "Fire Extinguisher Area" for det in detections)
    fire_extinguisher_detected = any(det["class"] == "Fire Extinguisher" for det in detections)
    door_detected = any(det["class"] == "Door" for det in detections)
    electrical_unit_detected = any(det["class"] == "Electrical Unit Area" for det in detections)
    electrical_unit_detected = any(det["class"] == "Electrical Unit" for det in detections)
    
    if obstruction_detected:
        if(washing_station_detected):
            violations.append("ANSI Z358.1-2014")
        if(fire_extinguisher_detected):
            violations.append("OSHA 1910.157(c)(1)")
        if(door_detected):
            violations.append("OSHA 1910.37(a)(3)")
        if(electrical_unit_detected):
            violations.append("OSHA 1910.303(g)(1)(i)(C)")
        
    print("Done detecting")
    return {"violations": violations}

@app.post("/testdetect")
async def testdetect(): 
    model = context["model"]
    
    
    # Get prediction from model
    results = model.predict(
        source="test1.jpg",  # Path to test image
        conf=0.4,  # Confidence threshold
    )
    
    # Get result
    result = results[0]
        
    # Get class names
    class_names = model.names
        
    violations = []
    detections = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        class_name = class_names[class_id]
        confidence = float(box.conf.item())
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
        "class": class_name,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2]
        })

        # Display Images with Bounding Boxes
        image = Image.open("test1.jpg")
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{class_name} {confidence:.2f}", fill="red")
        image.show()

    # Check if obstruction class was detected
    obstruction_detected = any(det["class"] == "Obstruction" for det in detections)
    washing_station_detected = any(det["class"] == "Washing Station Area" for det in detections)
    washing_station_detected = any(det["class"] == "Washing Station" for det in detections)
    fire_extinguisher_detected = any(det["class"] == "Fire Extinguisher Area" for det in detections)
    fire_extinguisher_detected = any(det["class"] == "Fire Extinguisher" for det in detections)
    door_detected = any(det["class"] == "Door" for det in detections)
    electrical_unit_detected = any(det["class"] == "Electrical Unit Area" for det in detections)
    electrical_unit_detected = any(det["class"] == "Electrical Unit" for det in detections)
    
    if obstruction_detected:
        if(washing_station_detected):
            violations.append("ANSI Z358.1-2014")
        if(fire_extinguisher_detected):
            violations.append("OSHA 1910.157(c)(1)")
        if(door_detected):
            violations.append("OSHA 1910.37(a)(3)")
        if(electrical_unit_detected):
            violations.append("OSHA 1910.303(g)(1)(i)(C)")
        
        
    return {"violations": violations, "detections": detections}

@app.get("/")
async def index():
    return "hello world"

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

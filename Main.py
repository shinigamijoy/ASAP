from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import torch
from transformers import BertTokenizer
import requests
# Load the NLU model
nlu_model = torch.load("/content/nlu_model.h5")

# Initialize tokenizer 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Function to validate Asset Number
def validate_asset_number(asset_number):
    if not 1 <= asset_number <= 50:
        raise ValueError("Asset Number must be between 1 and 50.")

# Function to validate Location
def validate_location(location):
    valid_locations = ["Computers Room", "Screens Room", "HeadPhones Room", "Mouses Room"]
    if location not in valid_locations:
        raise ValueError(f"Invalid location: {location}. Valid locations are: {', '.join(valid_locations)}.")

# Function to validate Quantity
def validate_quantity(quantity):
    if not 1 <= quantity <= 100:
        raise ValueError("Quantity must be between 1 and 100.")

# Function to validate Person
def validate_person(person):
    valid_persons = ["Ahmed", "Mohamed", "Aly", "Yasser"]
    if person not in valid_persons:
        raise ValueError(f"Invalid person: {person}. Valid persons are: {', '.join(valid_persons)}.")

# Function to perform Asset Move
def asset_move(asset_number, location_from, location_to, quantity):
    url = "https://task.asapsystems.com/AssetMove"
    payload = {
        "AssetNumber": asset_number,
        "LocationFrom": location_from,
        "LocationTo": location_to,
        "Quantity": quantity
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Function to perform Asset Receive
def asset_receive(asset_number, quantity, receive_location, person):
    url = "https://task.asapsystems.com/AssetReceive"
    payload = {
        "AssetNumber": asset_number,
        "Quantity": quantity,
        "ReceiveLocation": receive_location,
        "Person": person
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Function to perform Asset Adjust
def asset_adjust(asset_number, new_quantity, person, with_assignees):
    url = "https://task.asapsystems.com/AssetAdjust"
    payload = {
        "AssetNumber": asset_number,
        "NewQuantity": new_quantity,
        "Person": person,
        "WithAssignees": with_assignees
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

@app.post('/process_text')
async def process_text(request: Request):
    try:
        data = await request.json()
        text = data.get('text')  # Assuming the input text is provided in the JSON payload

        # Tokenize and preprocess the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Perform inference with the loaded NLU model
        outputs = nlu_model(**inputs)
        predicted_output = outputs['label']  # Assuming 'label' is the key for predicted action

        # Depending on the predicted output, perform the corresponding action
        if predicted_output == "Asset Move":
            asset_number = data["asset_number"]
            location_from = "Current Location"  # Assuming you can determine current location from your system
            location_to = data["location"]
            quantity = data["quantity"]
            validate_asset_number(asset_number)
            validate_location(location_from)
            validate_location(location_to)
            validate_quantity(quantity)
            result = asset_move(asset_number, location_from, location_to, quantity)
            logger.info(result)
        elif predicted_output == "Asset Receive":
            asset_number = data["asset_number"]
            quantity = data["quantity"]
            receive_location = data["location"]
            person = data["person"]  # Assuming you can determine the person from your system
            validate_asset_number(asset_number)
            validate_quantity(quantity)
            validate_location(receive_location)
            validate_person(person)
            result = asset_receive(asset_number, quantity, receive_location, person)
            logger.info(result)
        elif predicted_output == "Asset Adjust":
            asset_number = data["asset_number"]
            new_quantity = data["quantity"]
            person = "Ahmed"  # Assuming you can determine the person from your system
            with_assignees = False  # Depending on your system logic
            validate_asset_number(asset_number)
            validate_quantity(new_quantity)
            validate_person(person)
            result = asset_adjust(asset_number, new_quantity, person, with_assignees)
            logger.info(result)
        else:
            logger.error("Invalid action identified from NLU model output")

        return {"predicted_action": predicted_output, "message": "Request processed successfully."}

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(content={"error": "An error occurred while processing the request."}, status_code=500)
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
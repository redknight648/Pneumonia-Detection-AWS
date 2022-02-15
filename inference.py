import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Basic config for logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [JPEG_CONTENT_TYPE]  # Adding only support for jpeg images for now

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # calling gpu


def net():
    model=models.efficientnet_b4(pretrained=True)#load pretrained model
#     model = EfficientNet.from_pretrained('efficientnet-b4')

    for param in model.parameters():
        param.requires_grad = False

        # num_features = model.fc.in_features
    # model.fc =nn.Linear( num_features, 133)

    model.classifier = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625),  # 1792 is the orginal in_features
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2), )

    return model


# Overriding the model_fn to load our model
def model_fn(model_dir):
    logger.info("Inside model_fn function!")
    logger.info(f"Reading model from dir: {model_dir}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Starting to load the model...")
        model.load_state_dict(torch.load(f, map_location=device))
        logger.info("Successfully loaded the model")
    return model


# Overriding the default input_fn
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # Process an image uploaded to the endpoint
    logger.info(f'Incoming Requests Content-Type is: {content_type}')
    logger.info(f'Request body Type is: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        logger.info(f"Returning an image of type {content_type}")
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(
            f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {ACCEPTED_CONTENT_TYPE}")


# Overriding the default predict_fn
def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Starting the prediction process...")
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    logger.info("Starting to apply Transforms to the input image")
    input_object = test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda()  # put data into GPU
    logger.info("Completed applying Transforms to the input image")
    model.eval()
    with torch.no_grad():
        logger.info("Starting the model invokation")
        prediction = model(input_object.unsqueeze(0))
    return prediction
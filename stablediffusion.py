import boto3 # to use the AWS SDK for Python
import json # to work with JSON data
import base64 # to work with base64 encoded data
import os # to work with the operating system

# prompt_data="""
# Act as a Shakespeare and write a poem on Genertaive AI
# """


prompt_data = input("Enter the prompt: ") # take input as a string in AWS CLI

prompt_template = [{"text": prompt_data, "weight":1}] # to pass the prompt to the model so that it can generate the text based on the prompt
bedrock = boto3.client(service_name = "bedrock-runtime")
payload = {
    "text_prompts": prompt_template, # pass the prompt
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":512,
    "height":512
}

body = json.dumps(payload) # convert the payload to a JSON string
model_id = "stability.stable-diffusion-xl-v1"
response = bedrock.invoke_model( # invoke the model
    body = body, # pass the payload
    modelId = model_id, # pass the model id
    accept = "application/json", # accept JSON response from the model
    contentType = "application/json" # pass the payload
)

response_body = json.loads(response.get("body").read())
# print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# save the image to the disk in "output" folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
file_name = f"{output_folder}/generated_image.png"
with open(file_name, "wb") as file:
    file.write(image_bytes)
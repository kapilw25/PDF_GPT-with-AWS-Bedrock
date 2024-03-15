import boto3
import json

# prompt_data = """
# Act as a Shakespeare and write a poem on machine learning.
# """

# take input as a string in AWS CLI 
prompt_data = input("Enter the prompt: ")

bedrock = boto3.client(service_name='bedrock-runtime')

payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]", # [INST] and [/INST] are special tokens to indicate the start and end of the prompt
    "max_gen_len": 512, # maximum length of the generated text
    "temperature": 0.5, # temperature means randomness, 0 means deterministic, 1 means random
    "top_p": 0.9 # top_p means nucleus sampling, 0 means random, 1 means deterministic
}

body = json.dumps(payload) # convert the payload to a JSON string
model_id = "meta.llama2-70b-chat-v1" # model id of the model to be invoked
response = bedrock.invoke_model( # invoke the model
    body = body, # pass the payload
    modelId = model_id, # pass the model id
    accept = "application/json", # accept JSON response from the model
    contentType = "application/json" # pass the payload
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation'] # extract the generated text from the response
print(response_text)
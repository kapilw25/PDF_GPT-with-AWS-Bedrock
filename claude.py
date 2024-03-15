import boto3
import json


# prompt_data="""
# Act as a Shakespeare and write a poem on Genertaive AI
# """

# take input as a string in AWS CLI 
prompt_data = input("Enter the prompt: ")

# bedrock = boto3.client(service_name='bedrock-runtime')
bedrock=boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt": prompt_data,
    "maxTokens": 512, # maximum length of the generated text
    "temperature": 0.8, # temperature means randomness, 0 means deterministic, 1 means random
    "topP": 0.8 # top_p means nucleus sampling, 0 means random, 1 means deterministic
}


body = json.dumps(payload) # convert the payload to a JSON string
model_id = "ai21.j2-mid-v1"
# model_id = "anthropic.claude-v2"
response = bedrock.invoke_model( # invoke the model
    body = body, # pass the payload
    modelId = model_id, # pass the model id
    accept = "application/json", # accept JSON response from the model
    contentType = "application/json" # pass the payload
)

response_body = json.loads(response.get("body").read())
response_text = response_body.get('completions')[0].get('data').get('text')
print(response_text)

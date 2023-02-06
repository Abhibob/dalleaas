# dalleaas
DALLE as a service and with a Docker image

# Usage
## Docker Run Command
docker run \<OPTIONS> -e WANDB_API_KEY <YOUR_WANDB_API_KEY> -p \<OUTSIDE_PORT>:5000 --name \<NAME> dalle:latest

###  Port Mappings
Internally, the container runs a Flask app, so an outside port should be mapped to port 5000 to make it externally accessible. 
###  WanDB API Key
If the container is run in interactive terminal mode, the WanDB API Key can be entered through the terminal; however, in daemon mode it will not run if a WanDB API Key is specified. Specify a key in the container environment variables as indicated in the Docker Run Command. If you do not have one, create one by creating an account at https://wandb.ai. 
## HTTP Requests
Use the GET method at /generate-images. 
###  Body
Model:

    {
	    "prompts": ["prompt1", "prompt2"]
	    "image_number": 2
	}
	

   prompts is a list of prompts, image_number is an integer number of images

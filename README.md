# Binsense AI project


## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline 
7. Update the main.py
8. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/boruahbhaskar/binsenseai-ml
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p ./env python=3.8 -y
```

```bash
conda activate ./env
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python3 app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/boruahbhaskar/binsenseai-ml.mlflow \
MLFLOW_TRACKING_USERNAME= \
MLFLOW_TRACKING_PASSWORD= \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/bmsg7/binsenseai.mlflow

export MLFLOW_TRACKING_USERNAME=bmsg7 

export MLFLOW_TRACKING_PASSWORD=

```
# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 849369123332.dkr.ecr.us-east-1.amazonaws.com/binsenseai

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker

	check whether Docker have successfully installed or not by executing 
	
	docker --version
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

    # Download 

	# Create a folder
		$ mkdir actions-runner && cd actions-runner
	
	# Download the latest runner package
		$ curl -o actions-runner-linux-x64-2.316.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.316.1/actions-runner-linux-x64-2.316.1.tar.gz
 	
	# Optional: Validate the hash
		$ echo "d62de2400eeeacd195db91e2ff011bfb646cd5d85545e81d8f78c436183e09a8  actions-runner-linux-x64-2.316.1.tar.gz" | shasum -a 256 -c
		
	# Extract the installer
		$ tar xzf ./actions-runner-linux-x64-2.316.1.tar.gz



	## Configure 

	# Create the runner and start the configuration experience
		$ ./config.sh --url https://github.com/boruahbhaskar/binsenseai-ml --token AD3DCOLGTJLTRNLUTCL7NNTGJNRPC

	Once you paste the configure command and execute it will ask for a runner. Press enter and then write self-hosted. After few steps keep on pressing Enter 	
	
	# Last step, run it! It will connect EC2 with Github. Our Ci/CD is Set here.

		$ ./run.sh	

# 7. Setup github secrets:

	Setting AWS Secrets in Github
	Now We’ll need to Setup AWS Secrets with github.for that go Secrets & Variables & click on Actions.
	Click on new repository Secrets


    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  docker run -p 7860:7860  849369123332.dkr.ecr.us-east-1.amazonaws.com/binsenseai

    ECR_REPOSITORY_NAME = binsenseai



# 8. Start the EC2 instance and connect to EC2 through Terminal

   
		# Copy models to EC2 instance
		mkdir model


		cd to folder where key is located or  ~/Downloads/my-key-pair.pem 

		# chmod 400 /path/to/your-key-pair.pem


		scp -i "bb-aws-account.pem" ~/Machine_Learning/projects/model/resnet34_siamese_best.pth.tar ubuntu@ec2-34-207-108-195.compute-1.amazonaws.com:/home/ubuntu/model

		scp -i "bb-aws-account.pem" ~/Machine_Learning/projects/model/resnet50_best.pth.tar ubuntu@ec2-34-207-108-195.compute-1.amazonaws.com:/home/ubuntu/model


# 9. Run the docker instance using below command from EC2 instance

 docker run -p 7860:7860  849369123332.dkr.ecr.us-east-1.amazonaws.com/binsenseai

# 10. Access the App using the public IP address of EC2 instance . It should below format - 

 http://54.86.155.234:7860


# 11. Use the App for item verification and quantity validation  

## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model

# aic_transformer_8k
Download Google cloud sdk and install 
Create account in Google cloud platform
Create project with name - deploy-aic-lstm
Copy FastAPI project in local folder
Open COmmand Prompt with Admin rights
cd <folder path>
gcloud init
gcloud builds submit --tag gcr.io/deploy-aic-lstm/aiclstm 
Go to Cloud Run in services
Click on Create Service - Enter latest image and configurations as per project
Click on deploy
Click on 'Set Continuous deployment' and enter git hub repo so that any changes in git are pushed to cloud automatically

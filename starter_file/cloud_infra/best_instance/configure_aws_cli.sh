# filename: configure_aws_cli.sh
#!/bin/bash

# Extracting AWS credentials from the file
AWS_CREDS_FILE="/starter_file/cloud_infra/best_instance/.aws_creds"
AWS_ACCESS_KEY_ID=$(awk -F "=" '/aws_access_key_id/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')
AWS_SECRET_ACCESS_KEY=$(awk -F "=" '/aws_secret_access_key/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')
AWS_DEFAULT_REGION=$(awk -F "=" '/region/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set region $AWS_DEFAULT_REGION
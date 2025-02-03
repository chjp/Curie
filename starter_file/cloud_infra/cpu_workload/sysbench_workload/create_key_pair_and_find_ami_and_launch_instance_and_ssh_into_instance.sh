#!/bin/bash

# Step 1: Configure AWS CLI
source /starter_file/cloud_infra/best_instance/configure_aws_cli.sh

# Step 2: create a unique key pair
uuid=$(openssl rand -hex 16 | awk '{
    output = ""
    for (i = 1; i <= length($0); i += 2) {
        output = output substr($0, i, 2) "-"
    }
    sub(/-$/, "", output) # Remove the trailing hyphen
    print output
}') # Each key needs to be unique, otherwise we don't have access to it and SSH will fail. 
KEY_NAME="$uuid-key-pair"
KEY_PATH="$KEY_NAME.pem"
aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > $KEY_PATH

echo "Created key pair: $KEY_NAME"

# Set proper permissions for the key file
chmod 400 $KEY_PATH

# SECTION: Find the latest Amazon Linux 2 AMI ID. Do not use ????????, use * instead. 
AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-2.*-x86_64-gp2" --query "Images | sort_by(@, &CreationDate)[-1].ImageId" --output text)

if [ -z "$AMI_ID" ]; then
  echo "Failed to find valid AMI ID for Amazon Linux 2."
  exit 1
fi

# Output the AMI ID
echo "Found AMI ID: $AMI_ID"

# SECTION: Launch EC2 instance using key created in create_key_pair.sh
INSTANCE_TYPE="t2.micro"

INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --query 'Instances[0].InstanceId' --output text)

if [ -z "$INSTANCE_ID" ]; then
  echo "Failed to launch EC2 instance."
  exit 1
fi

# Output the instance ID
echo "Launched EC2 Instance ID: $INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Wait for the instance to be in running state
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Wait for instance to be ready: adjust as needed
sleep 20

# SECTION: SSH into instance:
ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$PUBLIC_IP 'curl -s ifconfig.me'
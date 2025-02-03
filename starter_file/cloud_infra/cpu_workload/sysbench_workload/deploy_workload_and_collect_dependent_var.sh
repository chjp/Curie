#!/bin/bash

# Step 1: Configure AWS CLI
AWS_CREDS_FILE="/starter_file/cloud_infra/cpu_workload/sysbench_workload/.aws_creds"
AWS_ACCESS_KEY_ID=$(awk -F = '/aws_access_key_id/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')
AWS_SECRET_ACCESS_KEY=$(awk -F = '/aws_secret_access_key/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')
AWS_DEFAULT_REGION=$(awk -F = '/region/ {print $2}' $AWS_CREDS_FILE | tr -d ' ')

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set region $AWS_DEFAULT_REGION

# Step 2: Create an EC2 Key Pair
uuid=$(openssl rand -hex 16 | awk '{
    output = ""
    for (i = 1; i <= length($0); i += 2) {
        output = output substr($0, i, 2) "-"
    }
    sub(/-$/, "", output) # Remove the trailing hyphen
    print output
}')
KEY_NAME="$uuid-key-pair"
KEY_PATH="$KEY_NAME.pem"
aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > $KEY_PATH
chmod 400 $KEY_PATH

echo "Created key pair: $KEY_NAME"

# Step 3: Find the Latest Amazon Linux 2 AMI
AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-2.*-x86_64-gp2" --query "Images | sort_by(@, &CreationDate)[-1].ImageId" --output text)

if [ -z "$AMI_ID" ]; then
  echo "Failed to find valid AMI ID for Amazon Linux 2."
  exit 1
fi

echo "Found AMI ID: $AMI_ID"

# Step 4: Launch an EC2 Instance
INSTANCE_TYPE="t3.micro"
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --query 'Instances[0].InstanceId' --output text)

if [ -z "$INSTANCE_ID" ]; then
  echo "Failed to launch EC2 instance."
  exit 1
fi

echo "Launched EC2 Instance ID: $INSTANCE_ID"

# Wait for the instance to be in running state
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get the public IP address of the instance
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Wait for the instance to be ready
sleep 20

# Step 5: Deploy and Test Web Service
ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$PUBLIC_IP << 'EOF'
# Install dependencies
sudo yum update -y

# Install stress-ng for traffic generation
sudo amazon-linux-extras install epel -y
sudo yum install -y sysbench

# Generate traffic and measure CPU utilization
sysbench cpu --cpu-max-prime=80000 run > results_metrics.txt
EOF

echo "Completed experiment run.."

# Retrieve CPU utilization results
scp -i $KEY_PATH ec2-user@$PUBLIC_IP:results_metrics.txt /workspace/results_<plan_id>_<group>_<partition_name>.txt

echo "Fetched and saved results.."

# Clean up
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

echo "Cleaned up resources.."
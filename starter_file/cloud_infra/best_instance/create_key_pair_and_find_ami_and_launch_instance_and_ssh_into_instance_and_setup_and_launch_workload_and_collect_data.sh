#!/bin/bash

# Step 1: Configure AWS CLI
source /starter_file/cloud_infra/best_instance/configure_aws_cli.sh

# Step 2: Create a unique key pair
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

# Set proper permissions for the key file
chmod 400 $KEY_PATH

echo "Key pair created: $KEY_NAME"
echo "Key pair saved to: $KEY_PATH"

# Step 3: Find the latest Amazon Linux 2 AMI ID
AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=amzn2-ami-hvm-2.*-x86_64-gp2" --query "Images | sort_by(@, &CreationDate)[-1].ImageId" --output text)

if [ -z "$AMI_ID" ]; then
  echo "Failed to find valid AMI ID for Amazon Linux 2."
  exit 1
fi

# Step 4: Launch an EC2 instance with the required type
INSTANCE_TYPE="c5.large" # Control group instance type
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --query 'Instances[0].InstanceId' --output text)

if [ -z "$INSTANCE_ID" ]; then
  echo "Failed to launch EC2 instance."
  exit 1
fi

# Wait for the instance to be in running state
echo "Waiting for the instance to run..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
sleep 20 # Allow additional time for the instance to be ready

echo "Instance launched: $INSTANCE_ID"
echo "Public IP address: $PUBLIC_IP"

# Step 5: Deploy the e-commerce application
ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$PUBLIC_IP <<'SSH_EOT'
cat <<'PYTHON_EOT' > ecommerce_app.py
from flask import Flask, jsonify, request

app = Flask(__name__)

products = [
    {"id": 1, "name": "Laptop", "price": 1200.0},
    {"id": 2, "name": "Smartphone", "price": 800.0},
    {"id": 3, "name": "Headphones", "price": 150.0},
]
cart = []

@app.route("/products", methods=["GET"])
def get_products():
    return jsonify(products), 200

@app.route("/cart", methods=["GET"])
def view_cart():
    total = sum(item["price"] for item in cart)
    return jsonify({"items": cart, "total": total}), 200

@app.route("/cart", methods=["POST"])
def add_to_cart():
    product_id = request.json.get("product_id")
    product = next((p for p in products if p["id"] == product_id), None)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    cart.append(product)
    return jsonify({"message": "Product added to cart"}), 200

@app.route("/checkout", methods=["POST"])
def checkout():
    total = sum(item["price"] for item in cart)
    cart.clear()
    return jsonify({"message": "Checkout successful", "total": total}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
PYTHON_EOT
pip3 install flask
sudo yum install httpd-tools -y
nohup python3 ecommerce_app.py &> app.log &

# Wait for the application to ensure it is running
sleep 10

# Check if the application is running
if ! pgrep -x "python3" > /dev/null; then
echo "Application failed to start" > app.log
exit 1
fi

# Step 6: Load testing
echo '{"product_id": 1}' > post_data.json
ab -n 500 -c 500 -p post_data.json -T application/json http://localhost:5000/cart > results.txt 2>&1
SSH_EOT

# Fetch the results and logs from the instance
scp -i $KEY_PATH ec2-user@$PUBLIC_IP:/home/ec2-user/results.txt /workspace/results_ba103c7d-143d-4b73-aaa6-d83ef1e79e25_control_group_partition_1.txt
scp -i $KEY_PATH ec2-user@$PUBLIC_IP:/home/ec2-user/app.log /workspace/app.log

# # Extract 99th percentile latency
# percentile_99th_latency=$(grep "99%" /home/ec2-user/results.txt | awk '{print $NF}')
# echo "99th Percentile Latency: $percentile_99th_latency ms" >> /workspace/results_ba103c7d-143d-4b73-aaa6-d83ef1e79e25_control_group_partition_1.txt

# Terminate instance after experiment
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID
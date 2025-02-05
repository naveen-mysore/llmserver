#############################
# Terraform + Provider Setup
#############################
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  # Region in which your key pair and instance should reside
  region = "us-west-2"
}

#############################
# EC2 Instance
#############################
resource "aws_instance" "app_server" {
  ami               = var.ami_id
  instance_type     = var.instance_type

  # Use the pre-existing AWS key pair created in the console
  key_name          = var.aws_key_pair_name

  # Place the instance into the custom subnet from network.tf
  subnet_id = aws_subnet.subnet_uno.id

  # Attach the security group from network.tf
  vpc_security_group_ids = [
    aws_security_group.ingress_all_test.id
  ]

  # User Data Script to Format and Mount EBS Volume
  user_data = <<-EOF
    #!/bin/bash
    set -ex

    # Wait for the volume to be available
    while [ ! -e /dev/xvdf ]; do sleep 1; done

    # Check if the volume is already formatted
    if ! blkid /dev/xvdf; then
      sudo mkfs -t ext4 /dev/xvdf
    fi

    # Create a mount directory and mount the volume
    sudo mkdir -p /mnt/data
    sudo mount /dev/xvdf /mnt/data

    # Persist the mount in /etc/fstab
    echo "/dev/xvdf /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

    # Change ownership so the default user has write access
    sudo chown ubuntu:ubuntu /mnt/data
  EOF

  tags = {
    Name = var.ami_name
  }
}

#############################
# Attach an Elastic IP (Optional)
#############################
resource "aws_eip" "ip_test_env" {
  instance = aws_instance.app_server.id
  vpc      = true
}

#############################
# Output SSH Instructions
#############################
output "ssh_instructions" {
  description = "How to SSH into the instance"
  value = <<EOT
To connect (assuming Amazon Linux 2):
  ssh -i ${var.local_private_key_path} ec2-user@${aws_eip.ip_test_env.public_ip}

If it's Ubuntu:
  ssh -i ${var.local_private_key_path} ubuntu@${aws_eip.ip_test_env.public_ip}

Adjust the username to match your AMI, and make sure your .pem file is chmod 400.
EOT
}
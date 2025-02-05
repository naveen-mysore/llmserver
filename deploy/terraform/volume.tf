#############################
# EBS Volume
#############################
resource "aws_ebs_volume" "app_server_volume" {
  availability_zone = "us-west-2a"  # Make sure this matches your instance's AZ
  size              = 50            # Set the size in GB (adjust as needed)
  type              = "gp3"         # General purpose SSD (gp3 is cost-effective)

  tags = {
    Name = "app-server-ebs-volume"
  }
}

#############################
# Attach the EBS Volume to the EC2 Instance
#############################
resource "aws_volume_attachment" "app_server_attach" {
  device_name = "/dev/xvdf"  # Linux device name (use /dev/sdf for Amazon Linux)
  volume_id   = aws_ebs_volume.app_server_volume.id
  instance_id = aws_instance.app_server.id
}
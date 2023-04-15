#!/bin/bash

aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 870481100706.dkr.ecr.ap-south-1.amazonaws.com
sudo docker build -t third-umpire-decision-automation .
sudo docker tag third-umpire-decision-automation 870481100706.dkr.ecr.ap-south-1.amazonaws.com/third-umpire-decision-automation:latest
sudo docker push 870481100706.dkr.ecr.ap-south-1.amazonaws.com/third-umpire-decision-automation:latest

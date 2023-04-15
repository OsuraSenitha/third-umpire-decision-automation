#!/bin/bash

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 870481100706.dkr.ecr.ap-southeast-1.amazonaws.com
docker build -t third-umpire-decision-automation .
docker tag third-umpire-decision-automation 870481100706.dkr.ecr.ap-southeast-1.amazonaws.com/third-umpire-decision-automation:latest
docker push 870481100706.dkr.ecr.ap-southeast-1.amazonaws.com/third-umpire-decision-automation:latest

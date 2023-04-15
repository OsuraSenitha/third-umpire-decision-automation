# third-umpire-decision-automation

Automate the third umpire decision in cricket using Machine Learning.

## Resource Utilization

1. Frontend: NextJS application. Hosted at [vercel.com](https://vercel.com).
2. Backend: Number of AWS services working together
    1. Data storage: AWS S3 Bucket (`arn:aws:s3:::third-umpire-decision-automation-osura`)
    2. API and Execution: AWS Lambda (`arn:aws:lambda:ap-south-1:870481100706:function:ThirdUmpireDecisionAutomationOsura`)
    3. Backend image builder: AWS EC2 (Instance ID: `i-090c0a61e03eee37e`)
    4. Backend image registry: AWS ECR (Image URI: `870481100706.dkr.ecr.ap-south-1.amazonaws.com/third-umpire-decision-automation:latest`)

## Deployment

1. Frontend: The `./frontend` directory contains a NextJS application. This is directly connected with a Vercel project. Hence, the deployment of the frontend leverages the luxury of CI/CD of vercel. So it was only required to push the commits to the github repository to reflect the changes in the final deployment.
2. Backend: The `./backend` directory contains the ML related backend code. The backend is developed as an AWS Lambda application and conterized into a docker image. This image was then published to the AWS Image Registry ECR. This image is then used to define the Lambda. To make the flow bandwidth efficient, the containerization and publish was done using an AWS t3-micro EC2 instance.

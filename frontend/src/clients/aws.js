import {
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
} from "@/constants/credentials";
import { LambdaClient } from "@aws-sdk/client-lambda";
import { S3Client } from "@aws-sdk/client-s3";

// Create an Amazon S3 service client object.
const s3Client = new S3Client({
    region: AWS_REGION,
    credentials: {
        accessKeyId: AWS_ACCESS_KEY_ID,
        secretAccessKey: AWS_SECRET_ACCESS_KEY,
    },
});

const lambdaClient = new LambdaClient({
    region: AWS_REGION,
    credentials: {
        accessKeyId: AWS_ACCESS_KEY_ID,
        secretAccessKey: AWS_SECRET_ACCESS_KEY,
    },
});

export { s3Client, lambdaClient };

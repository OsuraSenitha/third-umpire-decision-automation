import { InvokeCommand } from "@aws-sdk/client-lambda";
import { PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { lambdaClient, s3Client } from "@/clients/aws";
import {
    AWS_LAMBDA_NAME,
    AWS_S3_CLIENT_IMG_KEY,
    AWS_S3_NAME,
} from "@/constants/aws-resources";

export const handleUpload = async (lpImg) => {
    console.log("Uploading");
    const imgKey = AWS_S3_CLIENT_IMG_KEY + lpImg.name;
    const bucketParams = {
        Bucket: AWS_S3_NAME,
        Key: imgKey,
        Body: lpImg,
    };
    await s3Client.send(new PutObjectCommand(bucketParams));
    return imgKey;
};

export const handleInfer = async (imgKey) => {
    console.log("Infering");
    const payload = { imgKey };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    return result;
};

export const handleDownload = async ({ setImgSrc, imgS3Uri }) => {
    const content = imgS3Uri.substr(5, imgS3Uri.length).split("/");
    const bucketName = content[0];
    const imgKey = content.slice(1, content.length).join("/");
    const command = new GetObjectCommand({
        Bucket: bucketName,
        Key: imgKey,
    });

    try {
        const response = await s3Client.send(command);
        const buffer = await response.Body.transformToByteArray();
        const blob = new Blob([buffer]);
        const imgSrc = URL.createObjectURL(blob);
        if (setImgSrc) setImgSrc(imgSrc);
        return imgSrc;
    } catch (err) {
        console.error(err);
    }
};

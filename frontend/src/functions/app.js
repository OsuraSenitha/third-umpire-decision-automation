import { InvokeCommand } from "@aws-sdk/client-lambda";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { lambdaClient, s3Client } from "@/clients/aws";
import {
    AWS_LAMBDA_NAME,
    AWS_S3_CLIENT_IMG_KEY,
    AWS_S3_NAME,
} from "@/constants/aws-resources";

const handleUpload = async (lpImg) => {
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

const handleInfer = async (imgKey) => {
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

export const displayResults = (annotations, canvasRef, imgDim) => {
    for (let i = 0; i < annotations.length; i++) {
        const annotation = annotations[i];
        const [number, ...box] = annotation;
        annotate(canvasRef, box, number, imgDim);
    }
};

export const handleProcess = async ({
    img,
    canvasRef,
    imgDim,
    setProcessing,
    setAnnotations,
}) => {
    if (img) {
        setProcessing(true);
        const imgKey = await handleUpload(img);
        const lambdaRes = await handleInfer(imgKey);
        console.log(lambdaRes);
        const annotations = JSON.parse(lambdaRes).body.annotations;
        setAnnotations(annotations);
        displayResults(lambdaRes, canvasRef, imgDim);
        setProcessing(false);
    }
};

export const annotate = (canvasRef, box, number, imgDim) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.strokeStyle = "#3dfc03";
    context.lineWidth = 2;

    const [x1, y1, x2, y2] = box;

    const [x, y, w, h] = [x1, y1, x2 - x1, y2 - y1];
    const can_x = (x / imgDim[0]) * canvas.width;
    const can_y = (y / imgDim[1]) * canvas.height;
    const can_w = (w / imgDim[0]) * canvas.width;
    const can_h = (h / imgDim[1]) * canvas.height;

    // write number
    context.fillStyle = "#3dfc03";
    context.font = "lighter 8px Comic Sans MS";
    context.fillText(number, can_x, can_y - 2);

    context.strokeRect(can_x, can_y, can_w, can_h);
};

export const handleRemoveImg = ({ inputRef, setImg, setAnnotations }) => {
    setImg();
    setAnnotations([]);
    inputRef.current.value = "";
};

export const handleClear = (canvasRef) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
};

export const updateImageDim = (lpImg, setImgDim) => {
    var fr = new FileReader();
    fr.onload = function () {
        // file is loaded
        var img = new Image();
        img.onload = function () {
            setImgDim([img.width, img.height]);
        };
        img.src = fr.result; // is the data URL because called with readAsDataURL
    };
    fr.readAsDataURL(lpImg);
};

export const stringifyAnnotation = (annotation) => {
    const [number, x1, x2, y1, y2] = annotation;
    let [x, y, w, h] = [x1, y1, x1, x2, y1 - y2];
    [x, y, w, h] = [x, y, w, h].map((flt) => Math.round(flt));

    return `x: ${x}, y: ${y}, width: ${w}, height: ${h}, object key: ${number}`;
};

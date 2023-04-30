import { handleDownload, handleInfer, handleUpload } from "./aws";

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
    setResults,
}) => {
    if (img) {
        setProcessing(true);
        const imgKey = await handleUpload(img);
        const lambdaRes = await handleInfer(imgKey);
        console.log(lambdaRes);
        const parsed_res = JSON.parse(lambdaRes);
        if (parsed_res.statusCode == 200) {
            console.log(parsed_res.body);
            const batsman_img_src = await handleDownload({
                imgS3Uri: parsed_res.body.batsman_analysis_img_s3_uri,
            });
            console.log(parsed_res.body.wicket_s3_uri);
            const wicket_img_src = await handleDownload({
                imgS3Uri: parsed_res.body.wicket_s3_uri,
            });
            parsed_res.body.batsman_img_src = batsman_img_src;
            parsed_res.body.wicket_img_src = wicket_img_src;
            setResults(parsed_res.body);
            displayResults(lambdaRes, canvasRef, imgDim);
        } else {
            console.error(parsed_res.body);
        }
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

export const handleRemoveImg = ({ setImg, setResults, initResults }) => {
    setImg();
    setResults(initResults);
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

    return `${number}     x: ${x}, y: ${y}, width: ${w}, height: ${h}`;
};

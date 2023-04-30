import Head from "next/head";
import styles from "@/styles/pageStyles/index.js";
import { Box, Button, Typography, alpha, useTheme } from "@mui/material";
import {
    Close,
    OpenInNew,
    PlayCircleOutline,
    RestartAlt,
} from "@mui/icons-material";
import { useEffect, useRef, useState } from "react";
import {
    displayResults,
    handleClear,
    handleProcess,
    updateImageDim,
} from "@/functions/app";
import { LoadingButton } from "@mui/lab";
import { TEST_FILES_LINK } from "@/constants/routes";
import OutputRegion from "@/containers/outputRegion/OutputRegion";
import SelectedImage from "@/containers/inputRegion/selectedImage/SelectedImage";
import SelectImage from "@/containers/inputRegion/selectImage/SelectImage";
import commonStyles from "@/styles/commonStyles";
import InputRegion from "@/containers/inputRegion/InputRegion";

export default function Home() {
    const initResults = { annotations: [] };
    const [umpImg, setUmpImg] = useState();
    const [imgDim, setImgDim] = useState();
    const [processing, setProcessing] = useState(false);
    const [results, setResults] = useState(initResults);
    const canHeight = imgDim && parseInt((600 / imgDim[0]) * imgDim[1]) + "px";

    const canvasRef = useRef();
    const inputRef = useRef();

    useEffect(() => {
        if (umpImg) {
            handleClear(canvasRef);
            updateImageDim(umpImg, setImgDim);
        }
    }, [umpImg]);

    useEffect(() => {
        if (results.annotations.length > 0)
            displayResults(results.annotations, canvasRef, imgDim);
    }, [results]);
    return (
        <>
            <Head>
                <title>Third Umpire Decision Automation</title>
                <meta
                    name="description"
                    content="Automates the third umpires decisioning process"
                />
            </Head>
            <main style={styles.root}>
                <Box sx={styles.appRoot}>
                    <InputRegion
                        umpImg={umpImg}
                        setUmpImg={setUmpImg}
                        inputRef={inputRef}
                        canHeight={canHeight}
                        canvasRef={canvasRef}
                        setResults={setResults}
                        results={results}
                        initResults={initResults}
                    />
                    <Box sx={styles.btnContainer}>
                        <LoadingButton
                            onClick={() =>
                                handleProcess({
                                    img: umpImg,
                                    canvasRef,
                                    imgDim,
                                    setProcessing,
                                    setResults,
                                }).catch(console.error)
                            }
                            variant="contained"
                            loading={processing}
                            sx={commonStyles.btn}
                            endIcon={<PlayCircleOutline />}
                            disabled={
                                results.annotations.length !== 0 ||
                                umpImg === undefined
                            }
                        >
                            Process
                        </LoadingButton>
                        <Button
                            variant="contained"
                            onClick={() => handleClear(canvasRef)}
                            sx={commonStyles.btn}
                            endIcon={<RestartAlt />}
                            color="error"
                            disabled={results.annotations.length === 0}
                        >
                            Clear
                        </Button>
                    </Box>
                    <OutputRegion batsman_img_path="s3://third-umpire-decision-automation-osura/results/0b3ad937-0a46-4568-9e93-ec2fb015260a/batsman-analysis-img.jpg" />
                </Box>
            </main>
        </>
    );
}

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
    handleRemoveImg,
    stringifyAnnotation,
    updateImageDim,
} from "@/functions/app";
import { LoadingButton } from "@mui/lab";
import { TEST_FILES_LINK } from "@/constants/routes";

export default function Home() {
    const [umpImg, setUmpImg] = useState();
    const [imgDim, setImgDim] = useState();
    const [processing, setProcessing] = useState(false);
    const [annotations, setAnnotations] = useState([]);
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
        if (annotations.length > 0)
            displayResults(annotations, canvasRef, imgDim);
    }, [annotations]);
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
                    <a href={TEST_FILES_LINK} target="_blank">
                        <Button
                            sx={styles.btn}
                            variant="contained"
                            endIcon={<OpenInNew />}
                        >
                            Test Files
                        </Button>
                    </a>
                    <input
                        type="file"
                        onChange={(e) => setUmpImg(e.target.files[0])}
                        ref={inputRef}
                        hidden
                    />
                    {!umpImg && (
                        <Box sx={styles.fileSelect}>
                            <Button onClick={() => inputRef.current.click()}>
                                Choose File
                            </Button>
                            <Typography variant="body1">
                                {umpImg ? umpImg.name : "No file chosen"}
                            </Typography>
                        </Box>
                    )}
                    {umpImg && (
                        <Box>
                            <Box sx={styles.imgContainer}>
                                <img
                                    src={URL.createObjectURL(umpImg)}
                                    style={{
                                        width: "600px",
                                        height: canHeight,
                                    }}
                                />
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        width: "600px",
                                        height: canHeight,
                                        position: "absolute",
                                        left: 0,
                                        top: 0,
                                    }}
                                />
                                <Close
                                    sx={styles.closeIcon}
                                    onClick={() =>
                                        handleRemoveImg({
                                            setImg: setUmpImg,
                                            setAnnotations,
                                            inputRef,
                                        })
                                    }
                                />
                                {annotations.length !== 0 && (
                                    <Box sx={styles.resultPane}>
                                        <Typography variant="body1">
                                            Objects detected at
                                        </Typography>
                                        <ul>
                                            {annotations.map(
                                                (annotation, index) => (
                                                    <li key={index}>
                                                        {stringifyAnnotation(
                                                            annotation
                                                        )}
                                                    </li>
                                                )
                                            )}
                                        </ul>
                                    </Box>
                                )}
                            </Box>
                        </Box>
                    )}
                    <Box sx={styles.btnContainer}>
                        <LoadingButton
                            onClick={() =>
                                handleProcess({
                                    img: umpImg,
                                    canvasRef,
                                    imgDim,
                                    setProcessing,
                                    setAnnotations,
                                }).catch(console.error)
                            }
                            variant="contained"
                            loading={processing}
                            sx={styles.btn}
                            endIcon={<PlayCircleOutline />}
                            disabled={
                                annotations.length !== 0 || umpImg === undefined
                            }
                        >
                            Process
                        </LoadingButton>
                        <Button
                            variant="contained"
                            onClick={() => handleClear(canvasRef)}
                            sx={styles.btn}
                            endIcon={<RestartAlt />}
                            color="error"
                            disabled={annotations.length === 0}
                        >
                            Clear
                        </Button>
                    </Box>
                </Box>
            </main>
        </>
    );
}

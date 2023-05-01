import Head from "next/head";
import styles from "@/styles/pageStyles/index.js";
import { Box, Button } from "@mui/material";
import { PlayCircleOutline, RestartAlt } from "@mui/icons-material";
import { useContext, useEffect, useRef, useState } from "react";
import {
    displayResults,
    handleClear,
    handleProcess,
    updateImageDim,
} from "@/functions/app";
import { LoadingButton } from "@mui/lab";
import OutputRegion from "@/containers/outputRegion/OutputRegion";
import commonStyles from "@/styles/commonStyles";
import InputRegion from "@/containers/inputRegion/InputRegion";
import { modalContext } from "@/providers/modalProvider/ModalProvider";

export default function Home() {
    const { setNotificationText } = useContext(modalContext);
    const initResults = { annotations: [] };
    const [umpImg, setUmpImg] = useState();
    const [imgDim, setImgDim] = useState();
    const [processing, setProcessing] = useState(false);
    const [results, setResults] = useState(initResults);
    const canHeight = imgDim && parseInt((600 / imgDim[0]) * imgDim[1]) + "px";

    const canvasRef = useRef();
    const inputRef = useRef();
    const imgRef = useRef();

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
                        imgRef={imgRef}
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
                                    setNotificationText,
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
                        {/* <Button
                            variant="contained"
                            onClick={() => handleClear(canvasRef)}
                            sx={commonStyles.btn}
                            endIcon={<RestartAlt />}
                            color="error"
                            disabled={results.annotations.length === 0}
                        >
                            Clear
                        </Button> */}
                    </Box>
                    {(results.batsman_img_src || results.wicket_img_src) && (
                        <OutputRegion
                            batsmanImgSrc={results.batsman_img_src}
                            batsmanComment={results.batsman_comment}
                            wicketImgSrc={results.wicket_img_src}
                            wicketComment={results.wicket_comment}
                        />
                    )}
                </Box>
            </main>
        </>
    );
}

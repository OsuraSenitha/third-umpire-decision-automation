import { Box, Typography } from "@mui/material";
import styles from "./styles";
import { handleRemoveImg, stringifyAnnotation } from "@/functions/app";
import { Close } from "@mui/icons-material";

const SelectedImage = ({
    umpImg,
    canHeight,
    canvasRef,
    setUmpImg,
    setResults,
    results,
    initResults,
    inputRef,
}) => {
    return (
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
                            setResults,
                            initResults,
                        })
                    }
                />
                {results.annotations.length !== 0 && (
                    <Box sx={styles.resultPane}>
                        <Typography variant="body1">
                            Objects detected at
                        </Typography>
                        <ul>
                            {results.annotations.map((annotation, index) => (
                                <li key={index}>
                                    {stringifyAnnotation(annotation)}
                                </li>
                            ))}
                        </ul>
                    </Box>
                )}
            </Box>
        </Box>
    );
};

export default SelectedImage;

import { handleDownload } from "@/functions/aws";
import { Box, Button, Typography } from "@mui/material";
import { useEffect, useRef, useState } from "react";
import styles from "./styles";
import Expander from "@/components/expander/Expander";

const OutputRegion = ({
    batsmanImgSrc,
    batsmanComment,
    wicketImgSrc,
    wicketComment,
}) => {
    let out = "Unknown";
    let color = "orange";
    if (batsmanComment == "True" && wicketComment == "False") {
        out = "Not out";
        color = "red";
    }
    if (batsmanComment == "False" && wicketComment == "True") {
        out = "Out";
        color = "green";
    }

    return (
        <Box sx={styles.root}>
            <Typography sx={styles.comment} color={color} variant="body1">
                {out}
            </Typography>
            <Expander>
                <Box sx={styles.analysisContainer}>
                    <Typography sx={styles.heading} variant="h5">
                        Batsman Analysis
                    </Typography>
                    <Box sx={styles.analysisBody}>
                        {batsmanImgSrc && (
                            <img
                                style={{ width: "70vw" }}
                                src={batsmanImgSrc}
                            />
                        )}
                        <Typography sx={styles.comment} variant="body1">
                            Passed crease: {batsmanComment}
                        </Typography>
                    </Box>
                </Box>
                <Box sx={styles.analysisContainer}>
                    <Typography sx={styles.heading} variant="h5">
                        Wicket Analysis
                    </Typography>
                    <Box sx={styles.analysisBody}>
                        {wicketImgSrc && <img src={wicketImgSrc} />}
                        <Typography sx={styles.comment} variant="body1">
                            Wicket broken: {wicketComment}
                        </Typography>
                    </Box>
                </Box>
            </Expander>
        </Box>
    );
};

export default OutputRegion;

import { TEST_FILES_LINK } from "@/constants/routes";
import { OpenInNew } from "@mui/icons-material";
import { Box, Button } from "@mui/material";
import SelectImage from "./selectImage/SelectImage";
import SelectedImage from "./selectedImage/SelectedImage";
import commonStyles from "@/styles/commonStyles";

const InputRegion = ({
    umpImg,
    setUmpImg,
    inputRef,
    canHeight,
    canvasRef,
    setResults,
    imgRef,
    initResults,
}) => {
    return (
        <Box sx={{ textAlign: "center" }}>
            <a href={TEST_FILES_LINK} target="_blank">
                <Button
                    sx={commonStyles.btn}
                    variant="contained"
                    endIcon={<OpenInNew />}
                >
                    Test Files
                </Button>
            </a>
            {!umpImg && (
                <SelectImage
                    setUmpImg={setUmpImg}
                    inputRef={inputRef}
                    umpImg={umpImg}
                />
            )}
            {umpImg && (
                <SelectedImage
                    umpImg={umpImg}
                    canHeight={canHeight}
                    canvasRef={canvasRef}
                    setUmpImg={setUmpImg}
                    setResults={setResults}
                    initResults={initResults}
                    imgRef={imgRef}
                />
            )}
        </Box>
    );
};

export default InputRegion;

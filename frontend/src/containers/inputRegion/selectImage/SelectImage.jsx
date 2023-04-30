import { Box, Button, Typography } from "@mui/material";
import styles from "./styles";

const SelectImage = ({ setUmpImg, inputRef, umpImg }) => {
    return (
        <Box sx={styles.fileSelect}>
            <input
                type="file"
                onChange={(e) => setUmpImg(e.target.files[0])}
                ref={inputRef}
                hidden
            />
            <Button onClick={() => inputRef.current.click()}>
                Choose File
            </Button>
            <Typography variant="body1">
                {umpImg ? umpImg.name : "No file chosen"}
            </Typography>
        </Box>
    );
};

export default SelectImage;

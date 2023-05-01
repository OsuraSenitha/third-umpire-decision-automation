import { Box, Typography } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import styles from "./styles";

const ModalContainer = ({
    children,
    heading,
    handleClose,
    centerAlign,
    sx,
}) => {
    return (
        <Box sx={styles.backdrop} onClick={handleClose}>
            <Box sx={styles.modal} onClick={(e) => e.stopPropagation()}>
                <Box sx={styles.head}>
                    <Typography variant="h4">{heading}</Typography>
                    <CloseIcon sx={styles.cross} onClick={handleClose} />
                </Box>
                {centerAlign ? (
                    <Box sx={{ ...styles.body, ...sx }}>{children}</Box>
                ) : (
                    <Box sx={sx}>{children}</Box>
                )}
            </Box>
        </Box>
    );
};

export default ModalContainer;

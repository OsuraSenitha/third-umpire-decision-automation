import { Box, Typography } from "@mui/material";
import styles from "./styles";
import { useState } from "react";

const Expander = ({ children }) => {
    const [active, setActive] = useState(false);
    return (
        <Box sx={styles.root}>
            <Box sx={styles.head} onClick={() => setActive(!active)}>
                <Typography variant="caption">Analysis</Typography>
            </Box>
            <Box sx={styles.childContainer(active)}>{children}</Box>
        </Box>
    );
};

export default Expander;

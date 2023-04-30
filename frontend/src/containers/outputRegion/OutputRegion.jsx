import { Box, Typography } from "@mui/material";

const OutputRegion = ({ batsman_img_path }) => {
    return (
        <Box>
            <Box>
                <Typography variant="h3">Batsman Analysis</Typography>
                {batsman_img_path}
            </Box>
            <Box>
                <Typography variant="h3">Wicket Analysis</Typography>
            </Box>
        </Box>
    );
};

export default OutputRegion;

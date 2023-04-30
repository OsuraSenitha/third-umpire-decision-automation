import { alpha } from "@mui/system";

export default {
    imgContainer: {
        position: "relative",
        margin: "20px",
    },
    closeIcon: {
        position: "absolute",
        top: "0px",
        right: "0px",
        margin: "4px",
        padding: "2px",
        borderRadius: "3px",
        backgroundColor: (theme) => alpha(theme.palette.background.main, 0.2),
        ":hover": {
            cursor: "pointer",
            backgroundColor: (theme) =>
                alpha(theme.palette.background.main, 0.4),
        },
    },
    resultPane: {
        margin: "20px 0px",
    },
};

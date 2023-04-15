import { alpha } from "@mui/system";

export default {
    root: {
        maxWidth: "720px",
        margin: "auto",
        minHeight:"100vh"
    },
    appRoot: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        margin: "20px",
    },
    btn: {
        maxWidth: "180px",
        width: "fit-content",
        minWidth: "120px",
        margin: "0px 10px",
    },
    btnContainer: {
        display: "flex",
        margin: "20px",
    },
    fileSelect: {
        display: "flex",
        alignItems: "center",
        margin: "20px",
        border: "1px dashed gray",
        borderRadius: "5px",
        padding: "40px",
        minWidth: "650px",
        justifyContent: "center",
    },
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
}
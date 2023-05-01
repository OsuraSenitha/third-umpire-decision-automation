const styles = {
    root: {
        backgroundColor: (theme) => theme.palette.background.main,
        minHeight: "100vh",
        borderRadius: 0,
        "@media only screen and (max-width: 600px)": {
            position: "relative",
            overflowX: "hidden",
        },
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        overflow: "hidden",
        position: "relative",
    },
    contentRoot: {
        display: "flex",
        flexDirection: "row",
        justifyContent: "center",
        minHeight: "calc(100vh - 155px)",
    },
    bodyContainer: {
        backgroundColor: (theme) => theme.palette.background.main,
        width: "100%",
    },
    notification: {
        transition: "0.3s",
        backgroundColor: (theme) => theme.palette.background.main,
        width: "300px",
        height: "50px",
        position: "fixed",
        left: "20px",
        bottom: "-50px",
        borderRadius: "5px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        boxShadow: "0px 0px 10px #0005",
    },
    messageContainer: {
        margin: "0px 10px",
    },
    crossContainer: {
        borderRadius: "3px",
        margin: "0px 10px",
        height: "30px",
        ":hover": {
            backdropFilter: "brightness(1.5)",
            cursor: "pointer",
        },
        "& div": {
            width: "30px",
            height: "2px",
            backgroundColor: (theme) => theme.palette.text.primary,
        },
        "& div:nth-of-type(1)": {
            transform: "translateY(14px) rotate(-45deg)",
        },
        "& div:nth-of-type(2)": {
            transform: "translateY(12px) rotate(45deg)",
        },
    },
    modalHeading: {
        margin: "10px 0px",
        maxWidth: "70vw",
    },
    scrollBarHider: (height) => ({
        height: `${height}px`,
        width: "calc(70vw - 15px)",
        overflowX: "hidden",
        overflowY: "clip",
    }),
    modalBody: {
        margin: "10px 0px",
        width: "70vw",
        maxHeight: "50vh",
        overflowY: "scroll",
        backgroundColor: (theme) => theme.palette.primary.main,
        padding: "10px",
    },
    modalBtn: {
        margin: "10px 5px",
        width: "100px",
    },
};

export default styles;

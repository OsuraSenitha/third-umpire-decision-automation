const styles = {
    backdrop: {
        position: "fixed",
        width: "120vw",
        height: "100vh",
        // backgroundColor: "rgba(0,0,0,0.7)",
        backdropilter: "blur(2px)",
        left: "0px",
        top: "0px",
        zIndex: 10000000000000,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
    },
    modal: {
        backgroundColor: (theme) => theme.palette.background.main,
        borderRadius: "10px",
        boxShadow: "0px 0px 10px #0002",
        padding: "20px",
    },
    head: {
        width: "100%",
        position: "relative",
        display: "flex",
        justifyContent: "center",
    },
    cross: {
        position: "absolute",
        right: "0px",
        cursor: "pointer",
    },
    body: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minWidth: "600px",
    },
};

export default styles;

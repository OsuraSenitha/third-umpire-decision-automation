export default {
    root: {
        border: "1px dashed gray",
        borderRadius: "10px",
        overflow: "hidden",
    },
    head: {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        color: "gray",
        padding: "15px",
        cursor: "pointer",
        backgroundColor: (theme) => theme.palette.background.card,
    },
    childContainer: (active) => ({
        padding: active ? "20px" : "0px",
        height: active ? "none" : "0px",
    }),
};

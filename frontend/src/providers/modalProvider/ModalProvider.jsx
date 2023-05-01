import { Box, Button, Typography } from "@mui/material";

import styles from "./styles";
import { createContext, useEffect, useRef, useState } from "react";
import ModalContainer from "../../components/modalContainer/ModalContainer";

export const modalContext = createContext();

const ModalProvider = ({ children }) => {
    // notification
    const [notificationText, setNotificationText] = useState("");
    const [notificationStyles, setNotificationStyles] = useState(
        styles.notification
    );
    useEffect(() => {
        if (
            (notificationStyles.bottom === styles.notification.bottom) &
            (notificationText !== "")
        ) {
            setNotificationStyles({ ...styles.notification, bottom: "40px" });
            setTimeout(() => {
                setNotificationStyles(styles.notification);
            }, 3000);
            setTimeout(() => {
                setNotificationText("");
            }, 4000);
        }
    }, [notificationText]);

    // message modal
    const [messageModalContent, setMessageModalContent] = useState({
        title: "",
        message: "",
        buttons: undefined,
    });
    const [showMessageModal, setShowMessageModal] = useState(false);
    const closeMessageModal = () => {
        setMessageModalContent({
            title: "",
            message: "",
            buttons: undefined,
        });
        setShowMessageModal(false);
    };
    useEffect(() => {
        if (messageModalContent.message !== "") {
            setShowMessageModal(true);
        }
    }, [messageModalContent]);

    // confirmation modal
    const [confirmationModalContent, setConfirmationModalContent] = useState({
        title: "",
        message: "",
        onConfirm: () => {},
        valid: false,
        finally: () => {},
    });
    const [showConfirmationModal, setShowConfirmationModal] = useState(false);
    const closeConfirmationModal = () => {
        confirmationModalContent.finally && confirmationModalContent.finally();
        setConfirmationModalContent({
            title: "",
            message: "",
            onConfirm: () => {},
            valid: false,
            finally: () => {},
        });
        setShowConfirmationModal(false);
    };
    const handleModalConfirmation = () => {
        try {
            confirmationModalContent.onConfirm();
            closeConfirmationModal();
        } catch (err) {
            console.error(err);
        }
    };
    useEffect(() => {
        if (confirmationModalContent.message !== "") {
            setShowConfirmationModal(true);
        }
    }, [confirmationModalContent]);

    // freeze the height of the scrolbar hider
    const messageModalBodyRef = useRef();
    const confirmationModalBodyRef = useRef();
    const [messageModalBodyHeight, setMessageModalBodyHeight] = useState();
    const [confirmationModalBodyHeight, setConfirmationModalBodyHeight] =
        useState();
    const [updateModalHeight, setUpdateModalHeight] = useState(0);
    useEffect(() => {
        if (showConfirmationModal) {
            const confirmationModalBodyHeight =
                confirmationModalBodyRef.current.clientHeight;
            setConfirmationModalBodyHeight(confirmationModalBodyHeight);
        }
        if (showMessageModal) {
            const messageModalBodyHeight =
                messageModalBodyRef.current.clientHeight;
            setMessageModalBodyHeight(messageModalBodyHeight);
        }
    }, [showMessageModal, showConfirmationModal, updateModalHeight]);

    return (
        <modalContext.Provider
            value={{
                setNotificationText,
                setMessageModalContent,
                setConfirmationModalContent,
                updateModalHeight,
                setUpdateModalHeight,
                confirmationModalContent,
            }}
        >
            <Box sx={{ position: "relative" }}>
                {children}

                {/* Notifications modal */}
                <Box sx={notificationStyles}>
                    <Box sx={styles.messageContainer}>
                        <Typography variant="body2">
                            {notificationText}
                        </Typography>
                    </Box>
                    <Box
                        sx={styles.crossContainer}
                        onClick={() =>
                            setNotificationStyles(styles.notification)
                        }
                    >
                        <div />
                        <div />
                    </Box>
                </Box>

                {/* Message modal */}
                {showMessageModal ? (
                    <ModalContainer handleClose={closeMessageModal} centerAlign>
                        <Typography variant="h2" sx={styles.modalHeading}>
                            {messageModalContent.title}
                        </Typography>
                        <Box sx={styles.scrollBarHider(messageModalBodyHeight)}>
                            <Typography
                                variant="body1"
                                sx={styles.modalBody}
                                ref={messageModalBodyRef}
                            >
                                {messageModalContent.message}
                            </Typography>
                        </Box>
                        <Box sx={{ display: "flex" }}>
                            <Button
                                variant="contained"
                                sx={styles.modalBtn}
                                onClick={closeMessageModal}
                            >
                                Close
                            </Button>
                            {messageModalContent.buttons && [
                                ...messageModalContent.buttons,
                            ]}
                        </Box>
                    </ModalContainer>
                ) : null}

                {/* Confirmation modal */}
                {showConfirmationModal ? (
                    <ModalContainer
                        handleClose={closeConfirmationModal}
                        centerAlign
                    >
                        <Typography variant="h2" sx={styles.modalHeading}>
                            {confirmationModalContent.title}
                        </Typography>
                        <Box
                            sx={styles.scrollBarHider(
                                confirmationModalBodyHeight
                            )}
                        >
                            <Typography
                                variant="body1"
                                sx={styles.modalBody}
                                ref={confirmationModalBodyRef}
                            >
                                {confirmationModalContent.message}
                            </Typography>
                        </Box>
                        <Box>
                            <Button
                                variant="outlined"
                                sx={styles.modalBtn}
                                onClick={closeConfirmationModal}
                                color="error"
                            >
                                {confirmationModalContent.btnTexts
                                    ? confirmationModalContent.btnTexts[0]
                                    : "Cancel"}
                            </Button>
                            <Button
                                variant="contained"
                                onClick={handleModalConfirmation}
                                sx={styles.modalBtn}
                                disabled={
                                    confirmationModalContent.valid !==
                                        undefined &&
                                    !confirmationModalContent.valid
                                }
                            >
                                {confirmationModalContent.btnTexts
                                    ? confirmationModalContent.btnTexts[1]
                                    : "Save"}
                            </Button>
                        </Box>
                    </ModalContainer>
                ) : null}
            </Box>
        </modalContext.Provider>
    );
};

export default ModalProvider;

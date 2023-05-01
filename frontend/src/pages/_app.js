import ModalProvider from "@/providers/modalProvider/ModalProvider";
import "@/styles/globals.css";
import themeconfig from "@/styles/themeconfig";
import { ThemeProvider, createTheme } from "@mui/material";

export default function App({ Component, pageProps }) {
    const theme = createTheme(themeconfig);
    return (
        <ThemeProvider theme={theme}>
            <ModalProvider>
                <Component {...pageProps} />
            </ModalProvider>
        </ThemeProvider>
    );
}

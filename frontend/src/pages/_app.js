import "@/styles/globals.css";
import themeconfig from "@/styles/themeconfig";
import { ThemeProvider, createTheme } from "@mui/material";

export default function App({ Component, pageProps }) {
    const theme = createTheme(themeconfig);
    return (
        <ThemeProvider theme={theme}>
            <Component {...pageProps} />
        </ThemeProvider>
    );
}

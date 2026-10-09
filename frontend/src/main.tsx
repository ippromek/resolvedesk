import { LucideProvider } from "lucide-react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import { ToastProvider } from "./toast";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LucideProvider size={16} strokeWidth={1.75}>
      <ToastProvider>
        <App />
      </ToastProvider>
    </LucideProvider>
  </StrictMode>,
);

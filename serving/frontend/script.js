document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    
    const loadingUI = document.getElementById("loading");
    const resultsUI = document.getElementById("results-display");
    
    // UI Elements
    const predClass = document.getElementById("pred-class");
    const predConf = document.getElementById("pred-conf");
    const confBar = document.getElementById("conf-bar");
    
    const originalPreview = document.getElementById("original-preview");
    const heatmapOverlay = document.getElementById("heatmap-overlay");
    const btnXai = document.getElementById("btn-xai");
    
    const btnEval = document.getElementById("btn-eval");
    const evalCharts = document.getElementById("eval-charts");
    
    const btnReset = document.getElementById("btn-reset");
    const dropContent = document.getElementById("drop-content");
    const uploadPreview = document.getElementById("upload-preview");

    // Drag and Drop Events
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Reset UI State completely
    btnReset.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        dropContent.classList.remove("hidden");
        uploadPreview.classList.add("hidden");
        btnReset.classList.add("hidden");
        resultsUI.classList.add("hidden");
        loadingUI.classList.add("hidden");
        heatmapOverlay.classList.add("hidden");
    });

    // Toggle Evaluation Charts
    btnEval.addEventListener("click", () => {
        evalCharts.classList.toggle("hidden");
        if (evalCharts.classList.contains("hidden")) {
            btnEval.textContent = "View Model Training Metrics";
        } else {
            btnEval.textContent = "Hide Metrics";
        }
    });

    // Toggle Explainable AI Overlay
    btnXai.addEventListener("click", () => {
        if (heatmapOverlay.classList.contains("hidden")) {
            heatmapOverlay.classList.remove("hidden");
            btnXai.textContent = "Hide AI Reasoning";
            btnXai.style.background = "var(--accent)";
            btnXai.style.color = "var(--bg-main)";
        } else {
            heatmapOverlay.classList.add("hidden");
            btnXai.textContent = "Show AI Reasoning (Grad-CAM)";
            btnXai.style.background = "transparent";
            btnXai.style.color = "var(--accent)";
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file.");
            return;
        }

        // Reset UI Context
        heatmapOverlay.classList.add("hidden");
        btnXai.textContent = "Show AI Reasoning (Grad-CAM)";
        btnXai.style.background = "transparent";
        btnXai.style.color = "var(--accent)";
        
        const dropContent = document.getElementById("drop-content");
        const uploadPreview = document.getElementById("upload-preview");

        // Reveal the "Remove" button
        btnReset.classList.remove("hidden");

        // Show local preview instantly in the drop zone!
        const reader = new FileReader();
        reader.onload = (e) => {
            originalPreview.src = e.target.result; // For XAI overlay
            uploadPreview.src = e.target.result;   // For Drop Zone
            dropContent.classList.add("hidden");
            uploadPreview.classList.remove("hidden");
        };
        reader.readAsDataURL(file);

        // Prep Upload
        const formData = new FormData();
        formData.append("file", file);

        // Switch to Loading UI
        resultsUI.classList.add("hidden");
        loadingUI.classList.remove("hidden");

        // Transmit to FastAPI
        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail) });
            }
            return response.json();
        })
        .then(data => {
            // Apply Results smoothly
            setTimeout(() => {
                loadingUI.classList.add("hidden");
                resultsUI.classList.remove("hidden");

                predClass.textContent = data.prediction;
                predConf.textContent = data.confidence.toFixed(1);

                // CSS Styling colors
                if (data.prediction === "PNEUMONIA") {
                    predClass.className = "pred-text pred-pneumonia";
                    confBar.style.background = "var(--pneumonia-color)";
                } else {
                    predClass.className = "pred-text pred-normal";
                    confBar.style.background = "var(--normal-color)";
                }

                // Animate progress bar
                setTimeout(() => {
                    confBar.style.width = data.confidence + "%";
                }, 100);

                // Set XAI Image layer
                heatmapOverlay.src = data.heatmap;
                
            }, 800); // Artificial micro-delay for dramatic effect
        })
        .catch(error => {
            loadingUI.classList.add("hidden");
            alert("Analysis failed: " + error.message);
        });
    }
});

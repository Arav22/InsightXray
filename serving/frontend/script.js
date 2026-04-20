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
    
    const btnReport = document.getElementById("btn-report");
    const historyList = document.getElementById("history-list");

    let currentPredictionId = null;

    // Load History on Startup
    loadHistory();

    // Drag and Drop Events
    dropZone.addEventListener("click", () => fileInput.click());

    // ... (rest of drag events)

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
        btnReport.classList.add("hidden");
    });

    // Download Medical Report
    btnReport.addEventListener("click", () => {
        if (currentPredictionId) {
            window.open(`/api/report/${currentPredictionId}`, "_blank");
        }
    });

    function loadHistory() {
        fetch("/api/history")
            .then(res => res.json())
            .then(data => {
                if (data.length === 0) return;
                historyList.innerHTML = "";
                data.forEach(item => {
                    const card = document.createElement("div");
                    card.className = "history-item";
                    card.innerHTML = `
                        <img src="/history_images/${item.image_path}" alt="Scan">
                        <div class="history-meta">
                            <div>
                                <strong>${item.prediction}</strong><br>
                                <small>${new Date(item.timestamp * 1000).toLocaleDateString()}</small>
                            </div>
                            <a href="/api/report/${item.id}" target="_blank" class="history-report-link">PDF</a>
                        </div>
                    `;
                    historyList.appendChild(card);
                });
            });
    }

    function handleFile(file) {
        // ... (rest of initial handleFile logic)
        btnReset.classList.remove("hidden");
        btnReport.classList.add("hidden"); 

        // Show local preview instantly in the drop zone!
        // ... (existing reader logic)

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

                currentPredictionId = data.id;
                btnReport.classList.remove("hidden");

                predClass.textContent = data.prediction;
                // ... (rest of result display logic)
                heatmapOverlay.src = data.heatmap;
                
                // Refresh History
                loadHistory();
            }, 800); 
        })
        .catch(error => {
            loadingUI.classList.add("hidden");
            alert("Analysis failed: " + error.message);
        });
    }
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

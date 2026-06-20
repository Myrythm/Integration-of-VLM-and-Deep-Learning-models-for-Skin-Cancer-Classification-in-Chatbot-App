(function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const placeholder = document.getElementById("upload-placeholder");
    const previewWrapper = document.getElementById("preview-wrapper");
    const previewImg = document.getElementById("preview-img");
    const detectBtn = document.getElementById("detect-btn");
    const detectBtnText = document.getElementById("detect-btn-text");
    const loading = document.getElementById("loading");
    const resultCard = document.getElementById("result-card");
    let uploadedFile = null;
    let previewSrc = null;

    function handleFile(file) {
        if (!file) return;
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
            alert("Format tidak didukung. Gunakan JPG, PNG, atau WEBP.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert("File terlalu besar. Maksimal 10 MB.");
            return;
        }
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            previewSrc = e.target.result;
            previewImg.src = previewSrc;
            previewWrapper.classList.remove("hidden");
            placeholder.classList.add("hidden");
            detectBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener("change", function (e) { handleFile(e.target.files[0]); });

    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add("bg-teal-50", "dark:bg-teal-900/10");
    });
    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("bg-teal-50", "dark:bg-teal-900/10");
    });
    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("bg-teal-50", "dark:bg-teal-900/10");
        handleFile(e.dataTransfer.files[0]);
    });

    detectBtn.addEventListener("click", async function () {
        if (!uploadedFile) return;
        detectBtn.disabled = true;
        detectBtnText.textContent = "Memproses...";
        loading.classList.remove("hidden");
        resultCard.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", uploadedFile);

        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch { data = { detail: text || "Server error" }; }
            if (!res.ok) {
                throw new Error(data.detail || "Gagal memproses (" + res.status + ")");
            }
            const det = data.detection;
            const conf = (det.confidence * 100).toFixed(2);
            const level = det.confidence >= 0.8 ? "Tinggi" : det.confidence >= 0.5 ? "Sedang" : "Rendah";
            const levelColor = det.confidence >= 0.8 ? "text-green-600 dark:text-green-400" : det.confidence >= 0.5 ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400";

            document.getElementById("result-label").textContent = det.label;
            document.getElementById("result-confidence").textContent = conf + "%";
            document.getElementById("result-bar").style.width = conf + "%";
            const levelEl = document.getElementById("result-level");
            levelEl.textContent = level;
            levelEl.className = "text-xs font-medium text-right " + levelColor;
            document.getElementById("result-model").textContent = "Model: " + det.model_version;

            sessionStorage.setItem("detectionResult", JSON.stringify(data));
            sessionStorage.setItem("uploadedImage", previewSrc);

            resultCard.classList.remove("hidden");
            resultCard.scrollIntoView({ behavior: "smooth" });

            setTimeout(function () {
                window.location.href = "/chat";
            }, 1500);
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            loading.classList.add("hidden");
            detectBtn.disabled = false;
            detectBtnText.textContent = "Deteksi Gambar";
        }
    });
})();

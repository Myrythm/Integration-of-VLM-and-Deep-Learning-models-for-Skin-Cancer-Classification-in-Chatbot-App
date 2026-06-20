(function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("preview-img");
    const classifyBtn = document.getElementById("classify-btn");
    const resultDiv = document.getElementById("result");
    let selectedFile = null;

    function handleFile(file) {
        if (!file) return;
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
            alert("Format tidak didukung. Gunakan PNG, JPG, atau WEBP.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert("File terlalu besar. Maksimal 10 MB.");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
        classifyBtn.disabled = false;
    }

    fileInput.addEventListener("change", function (e) {
        handleFile(e.target.files[0]);
    });

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

    classifyBtn.addEventListener("click", async function () {
        if (!selectedFile) return;
        classifyBtn.disabled = true;
        classifyBtn.textContent = "Memproses...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Gagal mengklasifikasi");
            }
            const data = await response.json();
            const det = data.detection;
            const conf = (det.confidence * 100).toFixed(2);
            const level = det.confidence >= 0.8 ? "Tinggi" : det.confidence >= 0.5 ? "Sedang" : "Rendah";
            const levelColor = det.confidence >= 0.8 ? "text-green-700 dark:text-green-400" : det.confidence >= 0.5 ? "text-amber-700 dark:text-amber-400" : "text-red-700 dark:text-red-400";

            document.getElementById("result-label").textContent = det.label;
            document.getElementById("result-confidence").textContent = conf + "%";
            document.getElementById("result-bar").style.width = conf + "%";
            const levelEl = document.getElementById("result-level");
            levelEl.textContent = level;
            levelEl.className = "text-xs font-medium " + levelColor;
            document.getElementById("result-model").textContent = "Model: " + det.model_version;

            const chatUrl = "/chat?session=" + encodeURIComponent(data.chat_session_id) +
                "&label=" + encodeURIComponent(det.label) +
                "&confidence=" + encodeURIComponent(det.confidence);
            document.getElementById("chat-link").href = chatUrl;

            resultDiv.classList.remove("hidden");
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.textContent = "Klasifikasikan";
        }
    });
})();

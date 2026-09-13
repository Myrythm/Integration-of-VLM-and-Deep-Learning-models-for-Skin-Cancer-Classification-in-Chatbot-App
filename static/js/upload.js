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
    const errorCard = document.getElementById("error-card");
    const errorTitle = document.getElementById("error-title");
    const errorMessage = document.getElementById("error-message");
    const reuploadBtn = document.getElementById("reupload-btn");
    let uploadedFile = null;
    let previewSrc = null;

    function errorTitleFor(status, detail) {
        const d = (detail || "").toLowerCase();
        if (status === 400 && d.includes("lesi kulit")) return "Bukan Gambar Lesi Kulit";
        if (status === 400) return "Gambar Tidak Valid";
        if (status === 413) return "File Terlalu Besar";
        if (status === 503) return "Layanan Tidak Tersedia";
        return "Gagal Memproses Gambar";
    }

    function showError(status, detail) {
        errorTitle.textContent = errorTitleFor(status, detail);
        errorMessage.textContent = detail || "Terjadi kesalahan yang tidak diketahui.";
        resultCard.classList.add("hidden");
        errorCard.classList.remove("hidden");
        errorCard.scrollIntoView({ behavior: "smooth" });
    }

    function resetUpload() {
        uploadedFile = null;
        previewSrc = null;
        fileInput.value = "";
        previewImg.removeAttribute("src");
        previewWrapper.classList.add("hidden");
        placeholder.classList.remove("hidden");
        errorCard.classList.add("hidden");
        resultCard.classList.add("hidden");
        detectBtn.disabled = true;
        detectBtnText.textContent = "Deteksi Gambar";
    }

    reuploadBtn.addEventListener("click", function () {
        resetUpload();
        fileInput.click();
    });

    function handleFile(file) {
        if (!file) return;
        errorCard.classList.add("hidden");
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
        dropZone.classList.add("drag-active");
    });
    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-active");
    });
    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("drag-active");
        handleFile(e.dataTransfer.files[0]);
    });

    detectBtn.addEventListener("click", async function () {
        if (!uploadedFile) return;
        detectBtn.disabled = true;
        detectBtnText.textContent = "Memproses...";
        loading.classList.remove("hidden");
        resultCard.classList.add("hidden");
        errorCard.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", uploadedFile);

        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch { data = { detail: text || "Server error" }; }
            if (!res.ok) {
                showError(res.status, data.detail || "Gagal memproses (" + res.status + ")");
                return;
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
                document.body.style.opacity = "0";
                document.body.style.transition = "opacity 0.4s ease";
                setTimeout(function () {
                    window.location.href = "/chat";
                }, 400);
            }, 2000);
        } catch (err) {
            showError(0, "Tidak dapat terhubung ke server. Periksa koneksi Anda lalu coba lagi. / Could not reach the server. Check your connection and try again.");
        } finally {
            loading.classList.add("hidden");
            detectBtnText.textContent = "Deteksi Gambar";
            // Tombol deteksi sengaja dibiarkan nonaktif setelah error/proses;
            // aktif kembali otomatis saat pengguna memilih gambar baru (handleFile)
            // atau menekan "Unggah Ulang Gambar".
        }
    });
})();

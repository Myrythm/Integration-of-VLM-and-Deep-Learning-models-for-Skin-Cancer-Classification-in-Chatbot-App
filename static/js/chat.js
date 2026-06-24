(function () {
    const chatArea = document.getElementById("chat-area");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const promptBubbles = document.getElementById("prompt-bubbles");
    const contextBadge = document.getElementById("context-badge");
    const ctxText = document.getElementById("ctx-text");

    let isProcessing = false;
    let detectionData = null;
    let uploadedImage = null;

    try {
        detectionData = JSON.parse(sessionStorage.getItem("detectionResult"));
        uploadedImage = sessionStorage.getItem("uploadedImage");
    } catch (e) {}

    if (!detectionData) {
        window.location.href = "/upload";
        return;
    }

    const detection = detectionData.detection;
    const sessionId = detectionData.chat_session_id;

    if (contextBadge && ctxText) {
        contextBadge.classList.remove("hidden");
        contextBadge.classList.add("flex");
        ctxText.textContent = detection.label + " · " + (detection.confidence * 100).toFixed(1) + "%";
    }

    /* ───── Prompt recommendations ───── */

    const promptRecommendations = {
        "Karsinoma Sel Basal": [
            "Apa gejala umum Karsinoma Sel Basal?",
            "Seberapa berbahaya Karsinoma Sel Basal?",
            "Apa pengobatan untuk Karsinoma Sel Basal?",
            "Bagaimana cara mencegahnya?"
        ],
        "Karsinoma Sel Skuamosa": [
            "Apa gejala umum Karsinoma Sel Skuamosa?",
            "Seberapa berbahaya kondisi ini?",
            "Apa pengobatan yang tersedia?",
            "Apa penyebabnya?"
        ],
        "Melanoma": [
            "Bagaimana cara mengenali Melanoma?",
            "Apa gejala awal Melanoma?",
            "Seberapa berbahaya Melanoma?",
            "Apa pengobatan untuk Melanoma?"
        ],
        "Nevus": [
            "Apakah Nevus berbahaya?",
            "Cara membedakan Nevus dengan Melanoma?",
            "Kapan Nevus perlu diperiksa?",
            "Apa penyebab munculnya Nevus?"
        ]
    };

    const prompts = promptRecommendations[detection.label] || [
        "Apa itu " + detection.label + "?",
        "Bagaimana cara pengobatannya?",
        "Apakah kondisi ini berbahaya?",
        "Bagaimana cara pencegahannya?"
    ];

    prompts.forEach(function (prompt) {
        const btn = document.createElement("button");
        btn.className = "prompt-chip";
        btn.textContent = prompt;
        btn.addEventListener("click", function () {
            chatInput.value = prompt;
            handleSend();
        });
        promptBubbles.appendChild(btn);
    });

    displayInitialResult();

    /* ───── Detection card ───── */

    function displayInitialResult() {
        const conf = detection.confidence;
        const pct = (conf * 100).toFixed(2);
        const level = conf >= 0.8 ? "Tinggi" : conf >= 0.5 ? "Sedang" : "Rendah";
        const levelClass = conf >= 0.8 ? "confidence-high" : conf >= 0.5 ? "confidence-medium" : "confidence-low";

        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-start animate-fade-in-up";

        const row = document.createElement("div");
        row.className = "flex items-start gap-3 w-full max-w-[560px]";

        /* Bot avatar */
        const avatar = document.createElement("div");
        avatar.className = "bot-avatar";
        avatar.innerHTML = '<i class="bi bi-shield-check"></i>';

        /* Card */
        const card = document.createElement("div");
        card.className = "detection-card flex-1 shadow-sm";

        let imageHTML = "";
        if (uploadedImage) {
            imageHTML = `<img src="${uploadedImage}" alt="Uploaded skin image" class="detection-card-image" />`;
        }

        card.innerHTML = `
            <div class="p-5">
                <div class="flex items-start gap-4 mb-4">
                    ${imageHTML}
                    <div class="flex-1 min-w-0">
                        <h2 class="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400 dark:text-slate-500 mb-3">Hasil Klasifikasi</h2>
                        <div class="flex items-end justify-between gap-3">
                            <div class="min-w-0">
                                <p class="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-0.5">Diagnosis</p>
                                <p class="text-xl font-extrabold text-slate-900 dark:text-white leading-tight truncate">${detection.label}</p>
                            </div>
                            <div class="text-right flex-shrink-0">
                                <p class="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-0.5">Confidence</p>
                                <p class="text-xl font-extrabold text-teal-600 dark:text-teal-400 leading-tight">${pct}%</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="w-full bg-slate-200/60 dark:bg-slate-700/40 rounded-full h-2 mb-1.5">
                    <div class="h-2 rounded-full progress-bar-glow bg-gradient-to-r from-teal-500 to-cyan-400 dark:from-teal-400 dark:to-cyan-300 transition-all duration-1000 ease-out"
                         style="width: 0%" id="chat-result-bar"></div>
                </div>
                <div class="flex items-center justify-between">
                    <p class="text-[10px] font-mono text-slate-400 dark:text-slate-500">Model: EfficientNetB3-v1</p>
                    <p class="text-xs font-semibold ${levelClass}">${level}</p>
                </div>

                <div class="disclaimer-banner mt-3.5 p-3 rounded-lg flex items-start gap-2.5">
                    <i class="bi bi-exclamation-triangle-fill text-amber-500 dark:text-amber-400 text-xs flex-shrink-0 mt-0.5 animate-pulse"></i>
                    <div>
                        <p class="text-[10px] font-bold text-amber-800 dark:text-amber-300">Hasil AI Bisa Keliru</p>
                        <p class="text-[10px] text-amber-700 dark:text-amber-400 leading-relaxed mt-0.5">Konfirmasi hasil klasifikasi dengan dokter spesialis kulit untuk diagnosis pasti.</p>
                    </div>
                </div>
            </div>
        `;

        row.appendChild(avatar);
        row.appendChild(card);
        wrapper.appendChild(row);
        chatArea.appendChild(wrapper);
        scrollToBottom(true);

        /* Animate progress bar */
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const bar = document.getElementById("chat-result-bar");
                if (bar) bar.style.width = pct + "%";
            });
        });
    }

    /* ───── User message ───── */

    function appendUserMessage(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-end animate-slide-right";
        const inner = document.createElement("div");
        inner.className = "user-bubble text-sm";
        inner.textContent = text;
        wrapper.appendChild(inner);
        chatArea.appendChild(wrapper);
        scrollToBottom(true);
    }

    /* ───── Bot bubble ───── */

    function createBotBubble() {
        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-start animate-slide-left";

        const row = document.createElement("div");
        row.className = "flex items-start gap-3 max-w-[85%]";

        const avatar = document.createElement("div");
        avatar.className = "bot-avatar";
        avatar.innerHTML = '<i class="bi bi-chat-dots-fill" style="font-size:12px"></i>';

        const inner = document.createElement("div");
        inner.className = "bot-bubble";

        const text = document.createElement("div");
        text.className = "prose-chat text-sm text-slate-700 dark:text-slate-300";
        inner.appendChild(text);

        row.appendChild(avatar);
        row.appendChild(inner);
        wrapper.appendChild(row);
        chatArea.appendChild(wrapper);
        return text;
    }

    /* ───── Typing indicator ───── */

    function showTypingIndicator() {
        const wrapper = document.createElement("div");
        wrapper.id = "typing-indicator";
        wrapper.className = "flex justify-start animate-fade-in";

        const row = document.createElement("div");
        row.className = "flex items-start gap-3";

        const avatar = document.createElement("div");
        avatar.className = "bot-avatar";
        avatar.innerHTML = '<i class="bi bi-chat-dots-fill" style="font-size:12px"></i>';
        avatar.style.animation = "pulseGlow 2s infinite";

        const inner = document.createElement("div");
        inner.className = "bot-bubble";
        inner.innerHTML = '<span class="typing-dot text-teal-500"></span><span class="typing-dot text-teal-500"></span><span class="typing-dot text-teal-500"></span>';

        row.appendChild(avatar);
        row.appendChild(inner);
        wrapper.appendChild(row);
        chatArea.appendChild(wrapper);
        scrollToBottom(true);
    }

    function removeTypingIndicator() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
    }

    function scrollToBottom(force = false) {
        const threshold = 150;
        const isNearBottom = chatArea.scrollHeight - chatArea.clientHeight - chatArea.scrollTop < threshold;
        if (force || isNearBottom) {
            chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: force ? "smooth" : "auto" });
        }
    }

    function processText(text) {
        return DOMPurify.sanitize(marked.parse(text));
    }

    /* ───── Send handler ───── */

    async function handleSend() {
        const message = chatInput.value.trim();
        if (!message || isProcessing) return;

        isProcessing = true;
        sendBtn.disabled = true;
        chatInput.value = "";
        chatInput.style.height = "auto";

        appendUserMessage(message);
        showTypingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message,
                    detection: detection
                })
            });

            removeTypingIndicator();
            const textDiv = createBotBubble();
            let fullText = "";
            let citationDiv = null;

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    let data;
                    try { data = JSON.parse(line.slice(6)); } catch { continue; }

                    if (data.type === "token") {
                        fullText += data.content || "";
                        textDiv.innerHTML = processText(fullText);
                        scrollToBottom();
                    } else if (data.type === "citation" && data.citations) {
                        if (!citationDiv) {
                            citationDiv = document.createElement("div");
                            citationDiv.className = "citation-card";
                            textDiv.parentElement.appendChild(citationDiv);
                        }
                        let html = '<p class="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-1.5">Referensi</p>';
                        for (const c of data.citations) {
                            html += '<a href="' + c.url + '" target="_blank" class="citation-link">'
                                + '<span class="citation-number">' + c.number + '</span>'
                                + '<span>' + c.title + ' — <span class="text-slate-400 dark:text-slate-500">' + c.source + '</span></span>'
                                + '</a>';
                        }
                        citationDiv.innerHTML = html;
                    } else if (data.type === "blocked") {
                        textDiv.innerHTML = '<div class="flex items-start gap-2"><i class="bi bi-shield-exclamation text-amber-500 mt-0.5"></i><p class="text-amber-600 dark:text-amber-400 text-sm">' + (data.content || "") + "</p></div>";
                    } else if (data.type === "done") {
                        if (fullText) textDiv.innerHTML = processText(fullText);
                    } else if (data.type === "error") {
                        textDiv.innerHTML = '<div class="flex items-start gap-2"><i class="bi bi-exclamation-circle text-red-500 mt-0.5"></i><p class="text-red-500 text-sm">Error: ' + (data.content || "Unknown") + "</p></div>";
                    }
                }
            }
        } catch (err) {
            removeTypingIndicator();
            const textDiv = createBotBubble();
            textDiv.innerHTML = '<div class="flex items-start gap-2"><i class="bi bi-exclamation-circle text-red-500 mt-0.5"></i><p class="text-red-500 text-sm">Maaf, terjadi kesalahan: ' + err.message + "</p></div>";
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    /* ───── Event listeners ───── */

    sendBtn.addEventListener("click", handleSend);
    chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    chatInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 150) + "px";
    });
})();

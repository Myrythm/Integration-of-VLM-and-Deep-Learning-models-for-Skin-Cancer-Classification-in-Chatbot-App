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

    contextBadge.classList.remove("hidden");
    contextBadge.classList.add("flex");
    ctxText.textContent = detection.label + " · " + (detection.confidence * 100).toFixed(1) + "%";

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
        btn.className = "flex-shrink-0 px-3 py-1.5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-full text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-teal-50 dark:hover:bg-teal-900/20 hover:text-teal-600 dark:hover:text-teal-400 hover:border-teal-300 dark:hover:border-teal-700 transition-colors";
        btn.textContent = prompt;
        btn.addEventListener("click", function () {
            chatInput.value = prompt;
            handleSend();
        });
        promptBubbles.appendChild(btn);
    });

    displayInitialResult();

    function displayInitialResult() {
        const msg = "Hasil Deteksi:\n\n**Jenis:** " + detection.label + "\n**Tingkat Akurasi:** " + (detection.confidence * 100).toFixed(1) + "%\n\nAnda dapat menggunakan chatbot ini untuk berkonsultasi mengenai kondisi kulit Anda.\n\n**Segera kunjungi fasilitas kesehatan untuk pemeriksaan lebih lanjut.**";
        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-start animate-fade-in-up";

        const inner = document.createElement("div");
        inner.className = "max-w-[80%] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-tl-sm p-4 shadow-sm";

        if (uploadedImage) {
            const img = document.createElement("img");
            img.src = uploadedImage;
            img.className = "w-full max-w-[200px] rounded-xl mb-3";
            inner.appendChild(img);
        }

        const text = document.createElement("div");
        text.className = "prose-chat text-sm text-slate-700 dark:text-slate-300";
        text.innerHTML = DOMPurify.sanitize(marked.parse(msg));
        inner.appendChild(text);

        wrapper.appendChild(inner);
        chatArea.appendChild(wrapper);
        scrollToBottom();
    }

    function appendUserMessage(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-end animate-fade-in-up";
        const inner = document.createElement("div");
        inner.className = "max-w-[80%] bg-teal-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm shadow-sm";
        inner.textContent = text;
        wrapper.appendChild(inner);
        chatArea.appendChild(wrapper);
        scrollToBottom();
    }

    function createBotBubble() {
        const wrapper = document.createElement("div");
        wrapper.className = "flex justify-start animate-fade-in-up";
        const inner = document.createElement("div");
        inner.className = "max-w-[80%] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-tl-sm p-4 shadow-sm";
        const text = document.createElement("div");
        text.className = "prose-chat text-sm text-slate-700 dark:text-slate-300";
        inner.appendChild(text);
        wrapper.appendChild(inner);
        chatArea.appendChild(wrapper);
        return text;
    }

    function showTypingIndicator() {
        const wrapper = document.createElement("div");
        wrapper.id = "typing-indicator";
        wrapper.className = "flex justify-start animate-fade-in";
        const inner = document.createElement("div");
        inner.className = "bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm";
        inner.innerHTML = '<span class="typing-dot text-slate-400"></span><span class="typing-dot text-slate-400"></span><span class="typing-dot text-slate-400"></span>';
        wrapper.appendChild(inner);
        chatArea.appendChild(wrapper);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function processText(text) {
        return DOMPurify.sanitize(marked.parse(text));
    }

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
                            citationDiv.className = "mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 space-y-1";
                            textDiv.parentElement.appendChild(citationDiv);
                        }
                        let html = '<p class="text-xs font-semibold text-slate-400 dark:text-slate-500 mb-1">Referensi:</p>';
                        for (const c of data.citations) {
                            html += '<p class="text-xs"><a href="' + c.url + '" target="_blank" class="text-teal-600 dark:text-teal-400 hover:underline">[' + c.number + '] ' + c.title + ' — ' + c.source + '</a></p>';
                        }
                        citationDiv.innerHTML = html;
                    } else if (data.type === "blocked") {
                        textDiv.innerHTML = '<p class="text-amber-600 dark:text-amber-400">' + (data.content || "") + "</p>";
                    } else if (data.type === "done") {
                        if (fullText) textDiv.innerHTML = processText(fullText);
                    } else if (data.type === "error") {
                        textDiv.innerHTML = '<p class="text-red-500">Error: ' + (data.content || "Unknown") + "</p>";
                    }
                }
            }
        } catch (err) {
            removeTypingIndicator();
            const textDiv = createBotBubble();
            textDiv.innerHTML = '<p class="text-red-500">Maaf, terjadi kesalahan: ' + err.message + "</p>";
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

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

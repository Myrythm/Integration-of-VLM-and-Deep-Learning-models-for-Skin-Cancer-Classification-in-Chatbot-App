(function () {
    const chatArea = document.getElementById("chat-area");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    let isStreaming = false;

    chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    window.sendMessage = async function () {
        const message = chatInput.value.trim();
        if (!message || isStreaming) return;

        isStreaming = true;
        sendBtn.disabled = true;
        chatInput.value = "";

        const userBubble = document.createElement("div");
        userBubble.className = "flex justify-end";
        userBubble.innerHTML = '<div class="max-w-[80%] bg-teal-50 dark:bg-teal-900/20 rounded-lg px-4 py-2 text-sm"><p class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">User</p><p class="text-slate-800 dark:text-slate-200"></p></div>';
        userBubble.querySelector("p:last-child").textContent = message;
        chatArea.appendChild(userBubble);

        const assistantBubble = document.createElement("div");
        assistantBubble.className = "flex justify-start";
        assistantBubble.innerHTML = '<div class="max-w-[80%] bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-sm w-full"><p class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Asisten</p><div id="assistant-text" class="text-slate-800 dark:text-slate-200"></div><div id="assistant-citations" class="mt-2 space-y-1"></div></div>';
        chatArea.appendChild(assistantBubble);

        const textDiv = assistantBubble.querySelector("#assistant-text");
        const citationDiv = assistantBubble.querySelector("#assistant-citations");
        textDiv.innerHTML = "&#9608;";

        chatArea.scrollTop = chatArea.scrollHeight;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: window.CHAT_SESSION,
                    message: message,
                    detection: {
                        label: window.CHAT_LABEL,
                        confidence: window.CHAT_CONFIDENCE,
                        model_version: "EfficientNetB3-v1",
                    },
                }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const data = JSON.parse(line.slice(6));

                    if (data.type === "token") {
                        fullText += data.content || "";
                        textDiv.textContent = fullText + "\u2588";
                        chatArea.scrollTop = chatArea.scrollHeight;
                    } else if (data.type === "citation" && data.citations) {
                        let html = '<p class="text-xs font-medium text-slate-500 dark:text-slate-400">Referensi:</p>';
                        for (const c of data.citations) {
                            html += '<p class="text-xs"><a href="' + c.url + '" target="_blank" class="text-teal-600 dark:text-teal-400 hover:underline">[' + c.number + '] ' + c.title + ' — ' + c.source + '</a></p>';
                        }
                        citationDiv.innerHTML = html;
                    } else if (data.type === "blocked") {
                        textDiv.textContent = data.content || "";
                    } else if (data.type === "done") {
                        textDiv.textContent = fullText;
                    } else if (data.type === "error") {
                        textDiv.textContent = "Error: " + (data.content || "Unknown error");
                    }
                }
            }
        } catch (err) {
            textDiv.textContent = "Error: " + err.message;
        } finally {
            isStreaming = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    };
})();

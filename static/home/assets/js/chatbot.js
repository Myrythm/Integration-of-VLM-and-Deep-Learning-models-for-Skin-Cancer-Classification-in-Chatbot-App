document.addEventListener("DOMContentLoaded", () => {
    const sendButton = document.getElementById("send-button");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;

    // Mode default
    let isDarkMode = false;

    // Cek preferensi pengguna
    if (localStorage.getItem("theme") === "dark") {
        enableDarkMode();
    }

    // Toggle theme
    themeToggle.addEventListener("click", () => {
        isDarkMode = !isDarkMode;
        if (isDarkMode) {
            enableDarkMode();
            localStorage.setItem("theme", "dark");
        } else {
            enableLightMode();
            localStorage.setItem("theme", "light");
        }
    });

    function enableDarkMode() {
        body.classList.remove("light-mode");
        body.classList.add("dark-mode");
        themeToggle.textContent = "Light Mode";
    }

    function enableLightMode() {
        body.classList.remove("dark-mode");
        body.classList.add("light-mode");
        themeToggle.textContent = "Dark Mode";
    }

    // Kirim pesan saat klik tombol
    sendButton.addEventListener("click", sendMessage);

    // Kirim pesan saat tekan Enter
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === "") return;

        appendMessage("user", message);
        userInput.value = "";
        chatBox.scrollTop = chatBox.scrollHeight;

        fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage("bot", data.reply);
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            appendMessage("bot", "Maaf, terjadi kesalahan. Silakan coba lagi.");
            chatBox.scrollTop = chatBox.scrollHeight;
            console.error("Error:", error);
        });
    }

    function appendMessage(sender, text) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);

        const textDiv = document.createElement("div");
        textDiv.classList.add("message-text");
        textDiv.textContent = text;

        messageDiv.appendChild(textDiv);
        chatBox.appendChild(messageDiv);
    }
});

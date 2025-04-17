document.addEventListener('DOMContentLoaded', function() {
    const chatArea = document.getElementById('chat-area');
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const modeToggle = document.getElementById('mode-toggle');
    const loadingDiv = document.querySelector('.loading');
    const resultsDiv = document.getElementById('results');

    let isProcessing = false;
    const detectionResult = JSON.parse(sessionStorage.getItem('detectionResult'));
    const uploadedImage = sessionStorage.getItem('uploadedImage');

    if (!detectionResult || !uploadedImage) {
        window.location.href = '/upload'; // Redirect ke halaman upload jika tidak ada hasil deteksi
        return;
    }

    // Inisialisasi
    setupThemeMode();
    displayInitialResult();
    updatePromptBubbles(detectionResult.detections[0].label);

    async function handleUserInput() {
        if (isProcessing) return;

        const message = userInput.value.trim();
        if (!message) return;

        isProcessing = true;
        disableInput();

        appendUserMessage(message);
        userInput.value = '';
        
        showTypingIndicator();

        try {
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    detection_result: detectionResult
                })
            });

            const data = await response.json();
            appendBotMessage(data.reply);
        } catch (error) {
            appendBotMessage('Maaf, terjadi kesalahan dalam memproses permintaan Anda.');
        } finally {
            removeTypingIndicator();
            enableInput();
            isProcessing = false;
        }
    }

    function setupThemeMode() {
        const currentMode = localStorage.getItem('theme') || 'dark';
        setThemeMode(currentMode);

        modeToggle.addEventListener('click', () => {
            const newMode = document.body.classList.contains('light-mode') ? 'dark' : 'light';
            setThemeMode(newMode);
            localStorage.setItem('theme', newMode);
        });
    }

    function setThemeMode(mode) {
        document.body.className = mode === 'light' ? 'light-mode' : 'dark-mode';
        modeToggle.className = `bi ${mode === 'light' ? 'bi-sun' : 'bi-moon'} mode-toggle`;
    }

    function displayInitialResult() {
        const detectionMessage = `Hasil Deteksi:\nJenis: ${detectionResult.detections[0].label}\nTingkat Akurasi: ${(detectionResult.detections[0].confidence * 100).toFixed(1)}%\n\nAnda dapat menggunakan chatbot ini untuk berkonsultasi mengenai kondisi kulit Anda.\n\n<b>Segera kunjungi fasilitas kesehatan untuk pemeriksaan lebih lanjut.</b>`;
        const msgElement = appendBotMessageWithImage(detectionMessage, uploadedImage);
        chatArea.appendChild(msgElement);
        scrollToBottom();
    }

    // Fungsi untuk menambahkan pesan pengguna
    function appendUserMessage(text) {
        const msgDiv = createMessageElement('user', text);
        chatArea.appendChild(msgDiv);
        scrollToBottom();
    }

    // Fungsi untuk menambahkan pesan bot
    function appendBotMessage(text) {
        const msgDiv = createMessageElement('bot', text);
        chatArea.appendChild(msgDiv);
        scrollToBottom();
    }

    // Fungsi untuk menambahkan pesan bot dengan gambar
    function appendBotMessageWithImage(text, imageUrl) {
        const msgWrapper = document.createElement('div');
        msgWrapper.className = 'chat-message bot fade-in';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.style.textAlign = 'center'; 

        // Container untuk gambar
        const imgContainer = document.createElement('div');
        imgContainer.style.width = '100%';
        imgContainer.style.display = 'flex';
        imgContainer.style.justifyContent = 'start';
        imgContainer.style.marginBottom = '10px';

        // Tambahkan gambar jika ada
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.style.maxWidth = '200px';
            img.style.width = '100%';
            img.style.height = 'auto';
            img.style.borderRadius = '8px';
            img.style.objectFit = 'contain';
            imgContainer.appendChild(img);
        }

        content.appendChild(imgContainer);

        // Container untuk teks
        const textContainer = document.createElement('div');
        textContainer.style.textAlign = 'left'; // Ratakan teks ke kiri
        
        // Fungsi untuk mengganti **teks** dengan <strong>teks</strong>
        function replaceBold(text) {
            return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        }

        const processedText = replaceBold(text);
        const cleanHtml = DOMPurify.sanitize(processedText);
        textContainer.innerHTML = cleanHtml;
        content.appendChild(textContainer);

        msgWrapper.appendChild(content);
        return msgWrapper;
    }

    // Fungsi untuk memperbarui prompt bubbles
    function updatePromptBubbles(detectionLabel) {
        const promptContainer = document.querySelector('.prompt-scroll');
        promptContainer.innerHTML = '';
        
        const recommendations = promptRecommendations[detectionLabel] || [];
        
        recommendations.forEach(prompt => {
            const button = document.createElement('button');
            button.className = 'prompt-bubble';
            button.textContent = prompt;
            button.addEventListener('click', () => {
                userInput.value = prompt;
                handleUserInput();
            });
            promptContainer.appendChild(button);
        });
    }

    // Fungsi untuk mengganti **teks** dengan <strong>teks</strong> dan ### dengan <strong>
    function processText(text) {
        // Ganti teks yang dibungkus dengan ** ** menjadi <strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Ganti baris yang dimulai dengan ### menjadi <strong>
        text = text.replace(/^###\s*(.*?)$/gm, '<strong>$1</strong>');
        
        return text;
    }

    // Fungsi untuk membuat elemen pesan
    function createMessageElement(type, text) {
        const msgWrapper = document.createElement('div');
        msgWrapper.className = `chat-message ${type} fade-in`;
    
        const content = document.createElement('div');
        content.className = 'message-content';
    
        // Proses teks menggunakan fungsi processText
        const processedText = processText(text);
    
        // Bersihkan HTML menggunakan DOMPurify
        const cleanHtml = DOMPurify.sanitize(processedText);
    
        // Tetapkan HTML yang sudah dibersihkan ke elemen konten
        content.innerHTML = cleanHtml;
    
        msgWrapper.appendChild(content);
        return msgWrapper;
    }

    // Fungsi untuk menampilkan indikator mengetik
    function showTypingIndicator() {
        const indicator = createMessageElement('bot', '');
        indicator.id = 'typing-indicator';
        indicator.querySelector('.message-content').innerHTML = '<span class="typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
        chatArea.appendChild(indicator);
        scrollToBottom();
    }

    // Fungsi untuk menghapus indikator mengetik
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }

    // Fungsi untuk menggulir ke bawah
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // Fungsi untuk menonaktifkan input
    function disableInput() {
        userInput.disabled = true;
        sendBtn.disabled = true; // Menonaktifkan tombol kirim
        sendBtn.classList.add('disabled'); // Menambahkan kelas CSS jika diperlukan
    }

    // Fungsi untuk mengaktifkan input
    function enableInput() {
        userInput.disabled = false;
        sendBtn.disabled = false; // Mengaktifkan kembali tombol kirim
        sendBtn.classList.remove('disabled'); // Menghapus kelas CSS jika diperlukan
        userInput.focus();
    }

    // Event Listeners
    sendBtn.addEventListener('click', handleUserInput);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleUserInput();
        }
    });

    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = `${Math.min(this.scrollHeight, 150)}px`;
    });
});

// Define prompt recommendations 
const promptRecommendations = {
    'Karsinoma Sel Basal': [
        'Apa gejala umum Karsinoma Sel Basal?',
        'Seberapa berbahaya Karsinoma Sel Basal?',
        'Apa pengobatan untuk Karsinoma Sel Basal?',
        'Bagaimana cara mencegah Karsinoma Sel Basal?'
    ],
    'Karsinoma Sel Skuamosa': [
        'Apa gejala umum Karsinoma Sel Skuamosa?',
        'Seberapa Karsinoma Sel Skuamosa berbahaya?',
        'Apa pengobatan Karsinoma Sel Skuamosa?',
        'Apa penyebab Karsinoma Sel Skuamosa?'
    ],
    'Melanoma': [
        'Bagaimana cara mengenali Melanoma?',
        'Apa gejala awal Melanoma?',
        'Seberapa berbahaya Melanoma?',
        'Apa pengobatan untuk Melanoma?'
    ],
    'Nevus': [
        'Apakah Nevus berbahaya?',
        'Cara membedakan Nevus dengan melanoma?',
        'Kapan Nevus perlu diperiksa?',
        'Apa penyebab munculnya Nevus?'
    ]
};

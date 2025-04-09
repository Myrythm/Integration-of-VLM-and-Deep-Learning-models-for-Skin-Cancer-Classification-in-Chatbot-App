document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadSection = document.querySelector('.upload-preview');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const detectButton = document.getElementById('detectButton');
    const loadingDiv = document.querySelector('.loading');
    const modeToggle = document.getElementById('mode-toggle');
    const uploadPlaceholder = document.getElementById('upload-placeholder');

    let uploadedFile = null;

    // Theme Mode Setup
    setupThemeMode();
    
    // Upload Handlers Setup
    setupUploadHandlers();

    // Theme Mode Functions
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

    // Upload Handlers
    function setupUploadHandlers() {
        // Drag and drop events
        uploadSection.addEventListener('dragover', handleDragOver);
        uploadSection.addEventListener('dragleave', handleDragLeave);
        uploadSection.addEventListener('drop', handleDrop);
        
        // File input change event
        imageInput.addEventListener('change', handleFileSelect);
        
        // Detect button click event
        detectButton.addEventListener('click', handleDetection);
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        handleImageUpload(file);
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        handleImageUpload(file);
    }

    function handleImageUpload(file) {
        if (file && file.type.startsWith('image/')) {
            uploadedFile = file;
            const reader = new FileReader();
            
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadPlaceholder.style.display = 'none';
                detectButton.disabled = false;
                
                // Add animation class
                imagePreview.classList.add('fade-in');
            };
            
            reader.readAsDataURL(file);
        } else {
            showAlert('Format file tidak didukung. Gunakan file gambar (JPG, PNG, JPEG).', 'warning');
            resetUpload();
        }
    }

    async function handleDetection() {
        if (!uploadedFile) return;

        try {
            // Show loading state
            loadingDiv.style.display = 'flex';
            detectButton.disabled = true;

            // Create form data
            const formData = new FormData();
            formData.append('image', uploadedFile);

            // Send request to server
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                // Store detection result
                sessionStorage.setItem('detectionResult', JSON.stringify(result));
                sessionStorage.setItem('uploadedImage', imagePreview.src);
                
                // Show success message and redirect
                showAlert('Gambar berhasil diproses!', 'success').then(() => {
                    window.location.href = '/chatbot';
                });
            } else {
                // Handle specific error cases
                if (result.error.includes('bukan gambar kondisi kulit')) {
                    showAlert('Gambar yang diunggah bukan gambar kondisi kulit. Silakan unggah gambar yang sesuai.', 'warning');
                    resetUpload();
                } else {
                    showAlert(result.error, 'error').then(() => {
                        window.location.reload()
                    });
                    
                }
            }
        } catch (error) {
            showAlert('Terjadi kesalahan saat memproses gambar. Silakan coba lagi.', 'error');
        } finally {
            loadingDiv.style.display = 'none';
            detectButton.disabled = false;
        }
    }

    function resetUpload() {
        uploadedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        uploadPlaceholder.style.display = 'block';
        detectButton.disabled = true;
        
        // Add invalid animation
        uploadSection.classList.add('invalid');
        setTimeout(() => {
            uploadSection.classList.remove('invalid');
        }, 500);
    }

    function showAlert(message, type = 'error') {
        const config = {
            title: type === 'error' ? 'Error' 
                   : (type === 'warning' ? 'Peringatan' 
                   : (type === 'success' ? 'Sukses' : 'Info')),
            text: message,
            icon: type,
            iconColor: getIconColor(type),
            background: getBackgroundColor(type),
            confirmButtonText: 'OK',
            confirmButtonColor: getButtonColor(type),
            customClass: {
                popup: 'animated fadeInUp'
            },
            allowOutsideClick: true,
            allowEscapeKey: true
        };

        return Swal.fire(config);
    }

    // Utility functions for alert styling
    function getIconColor(type) {
        switch(type) {
            case 'success': return '#28a745';
            case 'info': return '#17a2b8';
            case 'warning': return '#ffc107';
            case 'error':
            default: return '#dc3545';
        }
    }

    function getBackgroundColor(type) {
        const isDark = document.body.classList.contains('dark-mode');
        const darkBg = '#ffffff';
        const lightBg = '#ffffff';
        
        return isDark ? darkBg : lightBg;
    }

    function getButtonColor(type) {
        switch(type) {
            case 'success': return '#28a745';
            case 'info': return '#17a2b8';
            case 'warning': return '#ffc107';
            case 'error':
            default: return '#088379';
        }
    }
});
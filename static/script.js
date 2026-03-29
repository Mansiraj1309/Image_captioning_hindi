document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("image-upload");

    // Allow clicking on the drop zone to open file dialog
    dropZone.addEventListener("click", () => fileInput.click());

    // Display image on conventional file selection
    fileInput.addEventListener("change", previewImage);

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    // Remove highlight when dragging leaves
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    document.getElementById("drop-zone").classList.add('dragover');
}

function unhighlight() {
    document.getElementById("drop-zone").classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files && files.length > 0) {
        document.getElementById("image-upload").files = files;
        previewImage();
    }
}

function previewImage() {
    const imageInput = document.getElementById("image-upload");
    const previewImage = document.getElementById("image-preview");
    const previewContainer = document.getElementById("image-preview-container");
    const dropZone = document.getElementById("drop-zone");
    const fileName = document.getElementById("file-name");

    if (imageInput.files && imageInput.files[0]) {
        const file = imageInput.files[0];
        // Validate image
        if (!file.type.startsWith('image/')) {
            alert("कृपया केवल चित्र (image file) अपलोड करें।");
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            fileName.textContent = file.name;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function uploadImage() {
    const imageInput = document.getElementById("image-upload");
    if (!imageInput.files || !imageInput.files[0]) {
        alert("कृपया एक चित्र चुनें या ड्रैग करें!");
        return;
    }

    // Switch to loading state
    document.getElementById("upload-form").style.display = 'none';
    document.getElementById("loading-container").style.display = 'block';

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide Loader
        document.getElementById("loading-container").style.display = 'none';

        if (data.error) {
            alert(data.error);
            resetForm();
            return;
        }

        // Show Results
        const resultContainer = document.getElementById("result-container");
        const resultImage = document.getElementById("result-image");
        const resultCaption = document.getElementById("result-caption");
        const audioSource = document.getElementById("audio-source");
        const captionAudio = document.getElementById("caption-audio");

        resultImage.src = data.image_path;
        resultCaption.textContent = data.caption;
        audioSource.src = data.audio_path;
        captionAudio.load();
        
        resultContainer.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('कैप्शन या ऑडियो जनरेट करने में त्रुटि हुई।');
        resetForm();
    });
}

function resetForm() {
    // Reset file input
    document.getElementById("image-upload").value = "";
    
    // Reset views
    document.getElementById("image-preview-container").style.display = 'none';
    document.getElementById("drop-zone").style.display = 'flex';
    document.getElementById("result-container").style.display = 'none';
    document.getElementById("loading-container").style.display = 'none';
    document.getElementById("upload-form").style.display = 'block';
    
    // Reset audio details
    const audioSource = document.getElementById("audio-source");
    const captionAudio = document.getElementById("caption-audio");
    audioSource.src = "";
    captionAudio.load();
}
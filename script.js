function previewImage() {
    const imageInput = document.getElementById("image-upload");
    const previewImage = document.getElementById("image-preview");
    const previewContainer = document.getElementById("image-preview-container");

    if (imageInput.files && imageInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(imageInput.files[0]);
    }
}

function uploadImage() {
    const imageInput = document.getElementById("image-upload");
    if (!imageInput.files[0]) {
        alert("कृपया एक चित्र चुनें!");
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        const resultContainer = document.getElementById("result-container");
        const resultImage = document.getElementById("result-image");
        const resultCaption = document.getElementById("result-caption");
        const audioSource = document.getElementById("audio-source");
        const captionAudio = document.getElementById("caption-audio");

        resultImage.src = data.image_path;
        resultCaption.textContent = data.caption;
        audioSource.src = data.audio_path;
        captionAudio.load(); // Reload audio element to update source
        resultContainer.style.display = 'block';
        document.getElementById("upload-form").style.display = 'none';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('कैप्शन या ऑडियो जनरेट करने में त्रुटि हुई।');
    });
}

function resetForm() {
    document.getElementById("image-upload").value = "";
    document.getElementById("image-preview").style.display = 'none';
    document.getElementById("image-preview-container").style.display = 'none';
    document.getElementById("upload-form").style.display = 'block';
    document.getElementById("result-container").style.display = 'none';
    const audioSource = document.getElementById("audio-source");
    const captionAudio = document.getElementById("caption-audio");
    audioSource.src = "";
    captionAudio.load();
}

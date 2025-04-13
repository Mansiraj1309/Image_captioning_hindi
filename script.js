function previewImage(event) {
    const imagePreview = document.getElementById("image-preview");
    const file = event.target.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
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

        resultImage.src = URL.createObjectURL(imageInput.files[0]);
        resultCaption.textContent = data.caption;
        resultContainer.style.display = 'block';
        document.getElementById("upload-form").style.display = 'none';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('कैप्शन जनरेट करने में त्रुटि हुई।');
    });
}

function resetForm() {
    document.getElementById("image-upload").value = "";
    document.getElementById("image-preview").style.display = 'none';
    document.getElementById("upload-form").style.display = 'block';
    document.getElementById("result-container").style.display = 'none';
}
// Initialization for ES User Interface

document.getElementById('searchType').addEventListener('change', function() {
    var searchType = this.value;
    var captionInput = document.getElementById('captionInput');
    var imageInput = document.getElementById('imageInput');
    // var imageInputDisplay = document.getElementById('imageInputDisplay');

    if (searchType === 'caption') {
        captionInput.style.display = 'block';
        imageInput.style.display = 'none';
        // imageInputDisplay.style.display = 'none';
    } else if (searchType === 'image') {
        captionInput.style.display = 'none';
        imageInput.style.display = 'block';
        // imageInputDisplay.style.display = 'block';
    }
});

document.getElementById('fileInput').addEventListener('change', function() {
    var file = this.files[0];
    var imagePreview = document.getElementById('imagePreview');
    var uploadedImage = document.getElementById('uploadedImage');

    if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
        }
        reader.readAsDataURL(file);
        imagePreview.style.display = 'block';
    } else {
        uploadedImage.src = '';
        imagePreview.style.display = 'none';
    }
});
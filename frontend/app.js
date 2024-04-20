//display different search type input form
document.getElementById('searchType').addEventListener('change', function() {
    var searchType = this.value;
    var captionInput = document.getElementById('captionInput');
    var imageInput = document.getElementById('imageInput');

    if (searchType === 'caption') {
        captionInput.style.display = 'block';
        imageInput.style.display = 'none';
    } else if (searchType === 'image') {
        captionInput.style.display = 'none';
        imageInput.style.display = 'block';
    }
});

//if uploaded, display search image
document.getElementById('fileInput').addEventListener('change', function() {
    var file = this.files[0];
    var imagePreview = document.getElementById('imagePreview');
    var uploadedImage = document.getElementById('uploadedImage');

    if (file) {
        var fileType = file.type.split('/')[0]; // Extract file type (e.g., 'image')
        if (fileType === 'image') {
            var reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
            }
            reader.readAsDataURL(file);
            imagePreview.style.display = 'block';
        }else{
            document.getElementById('fileInput').value = '';
            alert('Please upload an image file');
        }
    } else {
        uploadedImage.src = '';
        imagePreview.style.display = 'none';
    }
});

//display search results
// TODO
const displayResults = (data) => {

    const results = document.getElementById('results');
    results.innerHTML = '';

    if (data.length === 0) {
        results.innerHTML = 'No results found';
    } else {

        for(let i=0;i < data.length; i+=3){
            const row = document.createElement('div');
            row.className = 'row';

            for(let j=i;j<3;j++){
                if(j >= data.length){
                    break;
                }
                const item = data[j];

                const result = document.createElement('div');
                result.className = 'result col';

                const image = document.createElement('img');
                image.src = item;
                // image.alt = item.caption;
                image.className= "img-fluid img-size"

                result.appendChild(image);
                row.appendChild(result);
            }

            results.appendChild(col);
        }
    }

};

//search by caption send request
const searchByCaption = async (caption, index) => {
    const url = '/search_by_caption?caption=' + caption;// + '&topk=' + topK;
    console.log(url);
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            alert('caption uploaded successfully!');
        } else {
            alert('Error uploading caption.');
        }
    } catch (error) {
        alert('Error uploading caption: ' + error.message);
    }
    
    // const data = await response.json();
    // displayResults(data);
    // console.log(data);
};

//search by image send request
const searchByImage = async (file, index) => {
    const url = '/search_by_image';
    const formData = new FormData();
    formData.append('file', file);
    // formData.append('topk', topK);
    // console.log(url);
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            alert('File uploaded successfully!');
        } else {
            alert('Error uploading file.');
        }
    } catch (error) {
        alert('Error uploading file: ' + error.message);
    }
    // const data = await response.json();
    // displayResults(data);
    // console.log(data);
};

//search button click event
document.getElementById('submitSearch').addEventListener('click', function() {

    var searchType = document.getElementById('searchType').value;
    var index = document.getElementById('indexOption').value;
    switch (searchType) {
        case 'caption':
            var caption = document.getElementById('caption').value;
            if (caption) {
                // searchByCaption(caption, index);
                displayResults(["image-1.jpg","image-1.jpg","image-1.jpg","image-1.jpg"]);
                console.log("search by caption.");
            } else {
                alert('Please enter a caption');
            }
            break;
        case 'image':
            var file = document.getElementById('fileInput').files[0];
            if (file) {
                // searchByImage(file, index);
                displayResults(["image-1.jpg"]);
                console.log("search by image.");
            } else {
                alert('Please upload an image');
            }
            break;
        default:
            alert('Please select a search type');
    }
});

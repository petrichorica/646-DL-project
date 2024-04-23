const host_url = '127.0.0.1:5000';

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


const displayResults = (data) => {
    const results = document.getElementById('results');
    results.innerHTML = '';
    const container = $('#results');
    container.empty(); // Clear previous results


    if (data.length === 0) {
        results.innerHTML = '<p>No results found</p>'; 
    } else {
        for (let i = 0; i < data.length; i++) {
            const result = document.createElement('div');
            result.className = 'list';
            const item = data[i];
            const image = document.createElement('img');
            image.src = item;
            image.alt = 'Image';

            result.appendChild(image);
            container.append(result);
        }
        // for(let i = 0; i < data.length; i += 3){ 
        //     const row = document.createElement('div');
        //     row.className = 'row justify-content-center mb-3';

        //     for(let j = i; j < i + 3; j++){ 
        //         if(j >= data.length){
        //             break; 
        //         }

        //         const result = document.createElement('div');
        //         result.className = 'col-md-4 list'; 
        //         const item = data[j];
        //         const image = document.createElement('img');
        //         image.src = item;
        //         image.alt = 'Image';
        //         image.className = "img-fluid";

        //         result.appendChild(image);
        //         row.appendChild(result); 
        //     }

        //     results.appendChild(row); 
        // }
    }
};


function searchByCaption(caption, index) {
    const url = `http://${host_url}/search_by_caption?indexing=${index}`
    console.log("Sending POST request to:", url);

    $.ajax({
        url: url,
        type: 'POST',
        contentType: 'application/json', // Set the content type to application/json
        data: JSON.stringify({
            caption: caption
        }),
        dataType: 'json',
        success: function(data) {
            console.log("DATA:");
            console.log(data);
            console.log('Caption uploaded successfully!');
            // displayImages(data.images);
            displayResults(data.images);

        },
        error: function(jqXHR, textStatus, errorThrown) {
            console.log('Error uploading caption:', textStatus, errorThrown);
        }
    });
}

let uploadedFileObj = null;

async function searchByImage() {
    if (!uploadedFileObj) {
        alert('Please upload an image file');
        return;
    }
    const index = $("#indexOption").val();
    const url = `http://${host_url}/search_by_image?indexing=${index}`;
    const formData = new FormData();
    formData.append('file', uploadedFileObj);

    try {
        // Upload image to server
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            // alert('File uploaded successfully!');
            console.log("DATA:",data);
            displayResults(data.images);
            return data;
        } else {
            alert('Error uploading file.');
            return null;
        }
    } catch (error) {
        alert('Error uploading file: ' + error.message);
        return null;
    }
}

function setupImagePreview() {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadedImage = document.getElementById('uploadedImage');
    console.log("setupImagePreview");

    fileInput.addEventListener('change', (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        console.log("file" + file);
        if (file) {
            const fileType = file.type.split('/')[0];  // Extract file type
            if (fileType === 'image') {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
                uploadedFileObj = file;
    
            } else {
                fileInput.value = ''; // Reset file input
                uploadedFileObj = null;
                imagePreview.style.display = 'none';
                uploadedImage.src = '';
                alert('Please upload an image file');
            }
        }
    })
}

$(function() {
    setupImagePreview();
    // Rest of the onload code...
});


window.onload = function() {

    $("#searchType").change(function(){
        const searchType = $("#searchType").val();

        // Reset the file input and search result display
        document.getElementById('fileInput').value = '';
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('uploadedImage').src = '';
        uploadedFileObj = null;
        $("#results").empty();

        if(searchType === "image"){
            // Rebind the click event for the search button when the search type is 'image'
            $("#submitSearch").off('click').click(function(event) {
                event.preventDefault(); // Prevent the default form submission
                searchByImage();
            });
        } else {
            // Rebind the click event for the search button when the search type is 'caption'
            $("#submitSearch").off('click').click(function(event) {
                event.preventDefault(); // Prevent the default form submission
                searchCaption();
            });
        }
    });
};



function searchCaption() {
    const caption = $("#caption").val();
    const index = $("#indexOption").val();

     // Additional checks
     if (validate(caption) || validate(index)) {
        // Query creation endpoint
        searchByCaption(caption, index);
    } else{
        console.log("Not valid input");
    }
}

const validate=(element)=>{
    return !(element === "" || element == null);
};



function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    fetch(`http://${host_url}/save_file`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        alert('Image uploaded and saved successfully!');

    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error uploading file.');
    });
}


async function analyzeImage(){

let fileInput = document.getElementById("imageUpload");

if(fileInput.files.length === 0){
alert("Please upload an image");
return;
}

let formData = new FormData();
formData.append("image", fileInput.files[0]);

let response = await fetch("http://127.0.0.1:5000/predict",{
method:"POST",
body:formData
});

let data = await response.json();

localStorage.setItem("resultClass", data.class);
localStorage.setItem("resultConfidence", data.confidence);

window.location.href = "result.html";

}
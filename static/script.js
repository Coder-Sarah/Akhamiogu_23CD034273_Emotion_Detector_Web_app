let video, canvas, ctx;
const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('canvas');
const previewEl = document.getElementById('preview');
const resultEl = document.getElementById('result');

function startCam() {
    navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
        video = stream;
        videoEl.srcObject = stream;
        videoEl.play();
        videoEl.style.display = 'block';
        canvas = canvasEl.getContext('2d');
        canvasEl.width = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;
    });
}

function capturePhoto() {
    canvas.drawImage(videoEl, 0, 0);
    const dataUrl = canvasEl.toDataURL('image/jpeg');
    previewEl.src = dataUrl;
    previewEl.style.display = 'block';

    const blob = dataURLtoBlob(dataUrl);
    const formData = new FormData();
    formData.append('user_name', document.getElementById('userName').value);
    formData.append('image', blob, 'photo.jpg');

    fetch('/detect_emotion', {method: 'POST', body: formData})
        .then(res => res.json())
        .then(data => { 
            resultEl.textContent = data.emotion ?
                `Emotion: ${data.emotion}` :
                `Error: ${data.error}`;
        })
        .catch(err => resultEl.textContent = `Error: ${err}`);
}


// Existing file upload code (keep for image upload option)
document.getElementById('fileInput')?.addEventListener('change', /* your existing handler */);
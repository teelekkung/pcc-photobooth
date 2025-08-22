function switchCamera(port) {
    fetch('/set_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'camera_port=' + encodeURIComponent(port)
    }).then(res => console.log('Switched camera'));
}

function showLiveControls() {
    document.getElementById('live-controls').style.display = 'block';
    document.getElementById('capture-controls').style.display = 'none';
}

function showCaptureControls() {
    document.getElementById('live-controls').style.display = 'none';
    document.getElementById('capture-controls').style.display = 'block';
}

function captureImage() {
    const countdownDiv = document.createElement('div');
    countdownDiv.id = 'countdown';
    document.body.appendChild(countdownDiv);

    let count = 3;
    countdownDiv.innerText = count;
    countdownDiv.style.display = 'block';
    countdownDiv.style.opacity = '1';

    const interval = setInterval(() => {
        count--;
        if (count > 0) {
            countdownDiv.innerText = count;
        } else {
            clearInterval(interval);
            countdownDiv.innerText = '📸';
            setTimeout(() => {
                countdownDiv.remove();
                // Trigger capture after countdown
                fetch('/capture', { method: 'POST' })
                    .then(() => setTimeout(showCaptureControls, 500));
            }, 500);
        }
    }, 1000);
}

function returnLive() {
    fetch('/return_live', { method: 'POST' })
        .then(() => showLiveControls());
}

function downloadImage() {
    window.location.href = '/download';
}

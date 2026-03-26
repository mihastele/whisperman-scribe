let ws = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

const transcriptionModeSelect = document.getElementById('transcription-mode');
const modelTypeSelect = document.getElementById('model-type');
const modelSizeSelect = document.getElementById('model-size');
const languageSelect = document.getElementById('language');
const audioFileInput = document.getElementById('audio-file');
const uploadBtn = document.getElementById('upload-btn');
const startLiveBtn = document.getElementById('start-live-btn');
const stopLiveBtn = document.getElementById('stop-live-btn');
const liveStatusText = document.getElementById('live-status-text');
const outputText = document.getElementById('output-text');
const copyBtn = document.getElementById('copy-btn');
const saveBtn = document.getElementById('save-btn');
const clearBtn = document.getElementById('clear-btn');
const fileUploadSection = document.getElementById('file-upload-section');
const liveSection = document.getElementById('live-section');

async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        const data = await response.json();
        const languages = data.languages.sort();
        
        languages.forEach(lang => {
            const option = document.createElement('option');
            option.value = lang.toLowerCase();
            option.textContent = lang.charAt(0).toUpperCase() + lang.slice(1);
            languageSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load languages:', error);
    }
}

transcriptionModeSelect.addEventListener('change', (e) => {
    if (e.target.value === 'file') {
        fileUploadSection.style.display = 'block';
        liveSection.style.display = 'none';
    } else {
        fileUploadSection.style.display = 'none';
        liveSection.style.display = 'block';
    }
});

audioFileInput.addEventListener('change', (e) => {
    uploadBtn.disabled = !e.target.files.length;
});

uploadBtn.addEventListener('click', async () => {
    const file = audioFileInput.files[0];
    if (!file) return;
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Transcribing...';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelTypeSelect.value);
    formData.append('model_size', modelSizeSelect.value);
    
    const language = languageSelect.value;
    if (language) {
        formData.append('language', language);
    }
    
    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (outputText.textContent.trim()) {
            outputText.textContent += '\n\n';
        }
        outputText.textContent += data.text;
        
        updateOutputActions();
    } catch (error) {
        alert('Transcription failed: ' + error.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Transcribe File';
    }
});

async function startLiveTranscription() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/transcribe`);
        
        ws.onopen = async () => {
            ws.send(JSON.stringify({
                model_type: modelTypeSelect.value,
                model_size: modelSizeSelect.value,
                language: languageSelect.value || null
            }));
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    ws.send(event.data);
                }
            };
            
            mediaRecorder.start(250);
            isRecording = true;
            
            startLiveBtn.disabled = true;
            stopLiveBtn.disabled = false;
            liveStatusText.textContent = 'Listening...';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                console.error('WebSocket error:', data.error);
                return;
            }
            
            if (data.text && data.text.trim()) {
                if (outputText.textContent.trim()) {
                    outputText.textContent += ' ';
                }
                outputText.textContent += data.text;
                outputText.scrollTop = outputText.scrollHeight;
                updateOutputActions();
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            stopLiveTranscription();
        };
        
        ws.onclose = () => {
            if (isRecording) {
                stopLiveTranscription();
            }
        };
        
    } catch (error) {
        alert('Failed to start live transcription: ' + error.message);
    }
}

function stopLiveTranscription() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    isRecording = false;
    startLiveBtn.disabled = false;
    stopLiveBtn.disabled = true;
    liveStatusText.textContent = 'Ready to transcribe...';
}

startLiveBtn.addEventListener('click', startLiveTranscription);
stopLiveBtn.addEventListener('click', stopLiveTranscription);

function updateOutputActions() {
    const hasContent = outputText.textContent.trim().length > 0;
    copyBtn.disabled = !hasContent;
    saveBtn.disabled = !hasContent;
}

copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(outputText.textContent)
        .then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => {
                copyBtn.textContent = 'Copy';
            }, 2000);
        })
        .catch(err => {
            alert('Failed to copy text: ' + err.message);
        });
});

saveBtn.addEventListener('click', () => {
    const text = outputText.textContent.trim();
    if (!text) return;
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcription.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

clearBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the transcription?')) {
        outputText.textContent = '';
        updateOutputActions();
    }
});

loadLanguages();

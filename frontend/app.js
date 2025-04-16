
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('audioFile');
  const output = document.getElementById('output');

  if (fileInput.files.length === 0) {
    output.innerText = 'Por favor, selecione um ficheiro.';
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  output.innerHTML = 'A transcrever...';

  try {
    const response = await fetch('http://localhost:8000/transcribe', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    output.innerHTML = '<strong>Transcrição:</strong><br>' + data.transcription;
  } catch (error) {
    output.innerHTML = 'Erro ao transcrever.';
  }
});

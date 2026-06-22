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
    await OuviescreviAPI.init();
    const response = await fetch(`${OuviescreviAPI.getBase()}/transcribe`, {
      method: 'POST',
      body: formData,
      headers: OuviescreviAPI.authHeaders()
    });

    const data = await response.json();
    if (!response.ok) {
      output.innerHTML = data.detail || data.error || 'Erro ao transcrever.';
      return;
    }
    output.innerHTML = '<strong>Transcrição:</strong><br>' + data.transcription;
  } catch (error) {
    output.innerHTML = 'Erro ao transcrever.';
  }
});

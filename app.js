
document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('audioFile');
  const output = document.getElementById('output');

  if (fileInput.files.length === 0) {
    output.innerText = 'Por favor, selecione um ficheiro.';
    return;
  }

  const fileName = fileInput.files[0].name;
  output.innerHTML = '<strong>Ficheiro enviado:</strong> ' + fileName + '<br><br>' +
                     '<strong>Transcrição (simulada):</strong><br>' +
                     'Olá! Esta é uma transcrição simulada gerada automaticamente pela IA do TranscreveJá.';
});

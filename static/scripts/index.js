
const fileInput = document.getElementById('fileInput');
const displayArea = document.getElementById('displayArea');
const submitButton = document.getElementById('submitButton');
let processedText = '';

fileInput.addEventListener('change', function() {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(event) {
      let textContent = event.target.result;

     
      displayArea.classList.remove('hidden'); // Show textarea
      // Preprocessing
      displayArea.value = textContent;

       textContent = textContent.trim();
       textContent = textContent.replace(/\.([^\s])/g, '. $1');
       textContent = textContent.replace(/(?:^|[\.\!\?]\s+)([a-z])/g, (match) => match.toUpperCase());
       textContent = textContent.replace(/\s\s+/g, ' ');
      processedText = textContent;  // Store processed text
      submitButton.disabled = false;
    };
    reader.readAsText(file);
  }
});

submitButton.addEventListener('click', async function() {
  if (processedText) {
    try {
      // Send POST request to /upload route
      const response = await fetch('/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: processedText })
      });

      if (response.ok) {
        // Redirect to /result route after successful upload
        window.location.href = '/result';
      } else {
        alert('Failed to upload file. Please try again.');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    }
  }
});
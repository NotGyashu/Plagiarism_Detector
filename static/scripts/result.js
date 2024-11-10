// Function to preprocess the text for better readability
function preprocessText(text, plagiarizedSentences) {
    // Replace multiple spaces with a single space and trim the text
    let processedText = text.replace(/\s+/g, ' ').trim();

    // Highlight plagiarized sentences
    plagiarizedSentences.forEach(sentence => {
        const regex = new RegExp(`(${sentence})`, 'g'); // Create regex to find the sentence
        processedText = processedText.replace(regex, '<span class="plagiarism-text">$1</span>');
    });

    return processedText;
}

// Function to make the GET request to /check
async function fetchData() {
    const loader = document.getElementById('loader');
    const contentDisplay = document.getElementById('contentDisplay');
    const scoreDisplay = document.getElementById('scoreDisplay');
    const resultDiv = document.getElementById('result');

    // Show loading circle
    loader.style.display = 'block';

    try {
        const response = await fetch('/check');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        
        // Preprocess and display the content, highlighting plagiarized sentences
        const processedContent = preprocessText(data.content, data.plags);
        contentDisplay.innerHTML = processedContent; // Use innerHTML to render HTML
        contentDisplay.style.display = 'block'; // Show the content

        // Determine the plagiarism score class
        let scoreClass = '';
        const score = data.plagiarism_score;

        if (score < 30) {
            scoreClass = 'low';
        } else if (score < 70) {
            scoreClass = 'medium';
        } else if (score < 90) {
            scoreClass = 'high';
        } else {
            scoreClass = 'very-high';
        }

        // Display the score with appropriate class
        scoreDisplay.textContent = `Plagiarism Score: ${score}%`;
        scoreDisplay.className = `score ${scoreClass}`;
        resultDiv.style.display = 'block'; // Show the result
    } catch (error) {
        contentDisplay.textContent = 'Error fetching data: ' + error.message;
        scoreDisplay.textContent = ''; // Clear score display
        resultDiv.style.display = 'block'; // Show the error message
    } finally {
        // Hide loading circle
        loader.style.display = 'none';
    }
}

// Call the fetchData function when the page loads
window.onload = fetchData;
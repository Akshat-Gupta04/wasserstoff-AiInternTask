/**
 * AJAX Document Analysis Functionality for Document Research & Theme Identification Chatbot
 *
 * This script handles AJAX-based document uploads and analysis to prevent page reloads
 * when analyzing documents, preserving the session.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get the document upload form
    const uploadForm = document.querySelector('form[action*="index"]');
    const sessionId = new URLSearchParams(window.location.search).get('session_id');

    if (uploadForm && sessionId) {
        // Add event listener to the form
        uploadForm.addEventListener('submit', function(e) {
            // Prevent default form submission
            e.preventDefault();

            // Check if form is valid
            if (!this.checkValidity()) {
                // If not valid, trigger browser's native validation
                return false;
            }

            // Get form data
            const formData = new FormData(this);
            const queryInput = uploadForm.querySelector('textarea[name="query"]');
            const submitButton = document.getElementById('analyze-btn');
            const query = queryInput.value.trim();
            const fileInput = document.getElementById('file');

            // Don't submit if query is empty or no files selected
            if (!query) {
                alert('Please enter a question about your documents.');
                return false;
            }

            if (fileInput.files.length === 0) {
                alert('Please select at least one document to analyze.');
                return false;
            }

            // Show document processing animation
            window.showNewAnimation('document');

            // Disable submit button
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Analyzing...';
            }

            // Prepare the URL for the AJAX request
            const url = `/analyze/${sessionId}/ajax`;

            // Send AJAX request
            fetch(url, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Hide animation
                window.hideNewAnimation();

                if (data.success) {
                    // Redirect to chat page with the same session ID
                    window.location.href = `/chat/${sessionId}`;
                } else if (data.session_expired) {
                    // Show a brief message before reloading
                    alert('Your session has expired or been reset. The page will now reload to start fresh.');

                    // Reload the page immediately
                    window.location.reload();
                } else {
                    // Show error message
                    alert(data.error || 'An error occurred while analyzing your documents.');

                    // Re-enable submit button
                    if (submitButton) {
                        submitButton.disabled = false;
                        submitButton.innerHTML = '<i class="fas fa-search me-2"></i> Analyze Documents';
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);

                // Hide animation
                window.hideNewAnimation();

                // Show error message
                alert('An error occurred while analyzing your documents. Please try again.');

                // Re-enable submit button
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<i class="fas fa-search me-2"></i> Analyze Documents';
                }
            });

            return false;
        });
    }
});

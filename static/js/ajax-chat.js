/**
 * AJAX Chat Functionality for Document Research & Theme Identification Chatbot
 *
 * This script handles AJAX-based follow-up queries to prevent page reloads
 * when sending new messages in the chat interface.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get the chat form
    const chatForm = document.querySelector('.chat-input form');
    const chatMessages = document.getElementById('chat-messages');

    if (chatForm && chatMessages) {
        // Add event listener to the form
        chatForm.addEventListener('submit', function(e) {
            // Prevent default form submission
            e.preventDefault();

            // Get form data
            const formData = new FormData(this);
            const queryInput = chatForm.querySelector('textarea[name="query"]');
            const submitButton = chatForm.querySelector('button[type="submit"]');
            const query = queryInput.value.trim();

            // Don't submit if query is empty
            if (!query && !formData.get('file').size) {
                return;
            }

            // Add user message to chat immediately
            addUserMessage(query);

            // Clear the input field and reset its height
            queryInput.value = '';
            queryInput.style.height = 'auto';

            // Show loading indicator
            const loadingMessage = addLoadingMessage();

            // Disable submit button
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span>';
            }

            // Get the session ID from the form action URL
            let sessionId;

            // Try to extract from the form action URL
            if (chatForm.action.includes('?')) {
                sessionId = new URLSearchParams(chatForm.action.split('?')[1]).get('session_id');
            }

            // If not found, try to extract from the current URL
            if (!sessionId) {
                const urlParts = window.location.pathname.split('/');
                if (urlParts.length > 2 && urlParts[1] === 'chat') {
                    sessionId = urlParts[2];
                }
            }

            // If still not found, check if it's in the form as a data attribute
            if (!sessionId) {
                sessionId = chatForm.getAttribute('data-session-id');
            }

            // If still not found, check if it's in the page as a data attribute
            if (!sessionId) {
                sessionId = document.body.getAttribute('data-session-id');
            }

            console.log("Using session ID:", sessionId);

            // Store the session ID in localStorage to help with persistence
            if (sessionId) {
                localStorage.setItem('current_session_id', sessionId);
            }

            // Prepare the URL for the AJAX request
            // If sessionId is null or undefined, get it from the current URL
            if (!sessionId || sessionId === 'null' || sessionId === 'undefined') {
                const urlParts = window.location.pathname.split('/');
                if (urlParts.length > 2 && urlParts[1] === 'chat') {
                    sessionId = urlParts[2];
                    console.log("Extracted session ID from URL path:", sessionId);
                } else {
                    // Try to get it from the URL query parameters
                    const urlParams = new URLSearchParams(window.location.search);
                    sessionId = urlParams.get('session_id');
                    console.log("Extracted session ID from URL query:", sessionId);
                }
            }

            // Final check - if still null, redirect to home
            if (!sessionId || sessionId === 'null' || sessionId === 'undefined') {
                console.error("No valid session ID found. Redirecting to home page.");
                window.location.href = '/';
                return false;
            }

            const url = `/chat/${sessionId}/ajax`;

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
                // Remove loading message
                if (loadingMessage) {
                    loadingMessage.remove();
                }

                // Check if session expired
                if (data.session_expired) {
                    // Show a brief message before reloading
                    alert('Your session has expired or been reset. The page will now reload to start fresh.');

                    // Reload the page immediately
                    window.location.reload();
                }
                // Add AI response to chat
                else if (data.success) {
                    addAIMessage(data.message, data.metadata);
                } else {
                    addErrorMessage(data.error || 'An error occurred while processing your request.');
                }

                // Re-enable submit button (unless session expired)
                if (submitButton && !data.session_expired) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                }
            })
            .catch(error => {
                console.error('Error:', error);

                // Remove loading message
                if (loadingMessage) {
                    loadingMessage.remove();
                }

                // Add error message
                addErrorMessage('An error occurred while sending your message. Please try again.');

                // Re-enable submit button
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                }
            });
        });
    }

    // Function to add user message to chat
    function addUserMessage(message) {
        const timestamp = new Date().toLocaleTimeString();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message user-message mb-3 animate__animated animate__fadeIn';

        messageDiv.innerHTML = `
            <div class="message-header d-flex justify-content-between align-items-center mb-1">
                <div class="message-sender">
                    <i class="fas fa-user me-2"></i> <strong>You</strong>
                </div>
                <div class="message-time">
                    ${timestamp}
                </div>
            </div>
            <div class="message-bubble p-3 rounded">
                <div class="message-content">${message}</div>
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to add loading message
    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chat-message assistant-message mb-3 animate__animated animate__fadeIn loading-message';

        loadingDiv.innerHTML = `
            <div class="message-header d-flex justify-content-between align-items-center mb-1">
                <div class="message-sender">
                    <i class="fas fa-robot me-2"></i> <strong>AI Assistant</strong>
                </div>
                <div class="message-time">
                    ${new Date().toLocaleTimeString()}
                </div>
            </div>
            <div class="message-bubble p-3 rounded">
                <div class="message-content">
                    <div class="d-flex align-items-center">
                        <div class="spinner-border spinner-border-sm me-2" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <span>Analyzing your question...</span>
                    </div>
                </div>
            </div>
        `;

        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return loadingDiv;
    }

    // Function to add AI message to chat
    function addAIMessage(message, metadata) {
        const timestamp = new Date().toLocaleTimeString();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message assistant-message mb-3 animate__animated animate__fadeIn';

        let messageHTML = `
            <div class="message-header d-flex justify-content-between align-items-center mb-1">
                <div class="message-sender">
                    <i class="fas fa-robot me-2"></i> <strong>AI Assistant</strong>
                </div>
                <div class="message-time">
                    ${timestamp}
                </div>
            </div>
            <div class="message-bubble p-3 rounded">
                <div class="message-content">${message}</div>
        `;

        // Add tabs if metadata is available
        if (metadata && (metadata.themes || metadata.extracted_answers || metadata.graph_path)) {
            const tabId = 'tabs-' + Date.now();

            messageHTML += `
                <div class="result-section mt-3">
                    <ul class="nav nav-tabs result-tabs" role="tablist">
            `;

            if (metadata.themes && metadata.themes.length > 0) {
                messageHTML += `
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#themes-${tabId}" type="button" role="tab">
                            <i class="fas fa-lightbulb me-1"></i> Themes
                        </button>
                    </li>
                `;
            }

            if (metadata.extracted_answers && metadata.extracted_answers.length > 0) {
                messageHTML += `
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#answers-${tabId}" type="button" role="tab">
                            <i class="fas fa-table me-1"></i> Answers
                        </button>
                    </li>
                `;
            }

            if (metadata.graph_path) {
                messageHTML += `
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#graph-${tabId}" type="button" role="tab">
                            <i class="fas fa-project-diagram me-1"></i> Graph
                        </button>
                    </li>
                `;
            }

            messageHTML += `
                    </ul>
                    <div class="tab-content">
            `;

            // Themes tab
            if (metadata.themes && metadata.themes.length > 0) {
                messageHTML += `
                    <div class="tab-pane fade show active" id="themes-${tabId}" role="tabpanel">
                        <div class="row">
                `;

                metadata.themes.forEach((theme, index) => {
                    messageHTML += `
                        <div class="col-md-6 mb-2">
                            <div class="theme-card p-2 rounded animate__animated animate__fadeIn" data-index="${index}">
                                <i class="fas fa-tag me-2"></i> ${theme}
                            </div>
                        </div>
                    `;
                });

                messageHTML += `
                        </div>
                    </div>
                `;
            }

            // Answers tab
            if (metadata.extracted_answers && metadata.extracted_answers.length > 0) {
                messageHTML += `
                    <div class="tab-pane fade" id="answers-${tabId}" role="tabpanel">
                        <div class="table-container">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>File</th>
                                        <th>Answer</th>
                                        <th>Citation</th>
                                    </tr>
                                </thead>
                                <tbody>
                `;

                metadata.extracted_answers.forEach(answer => {
                    messageHTML += `
                        <tr>
                            <td>${answer.Filename}</td>
                            <td>${answer.Extracted_Answer}</td>
                            <td>${answer.Citation}</td>
                        </tr>
                    `;
                });

                messageHTML += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            // Graph tab
            if (metadata.graph_path) {
                messageHTML += `
                    <div class="tab-pane fade" id="graph-${tabId}" role="tabpanel">
                        <div class="position-relative">
                            <iframe class="graph-container" style="height: 400px;" src="/static/${metadata.graph_path}"></iframe>
                            <button class="btn btn-sm btn-primary fullscreen-btn" data-bs-toggle="tooltip" data-bs-placement="top" title="View fullscreen">
                                <i class="fas fa-expand"></i>
                            </button>
                        </div>
                    </div>
                `;
            }

            messageHTML += `
                    </div>
                </div>
            `;
        }

        messageHTML += `
            </div>
        `;

        messageDiv.innerHTML = messageHTML;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Initialize tooltips and tabs
        const tooltips = messageDiv.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip);
        });

        // Add event listeners to fullscreen buttons
        const fullscreenBtns = messageDiv.querySelectorAll('.fullscreen-btn');
        fullscreenBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const iframe = this.closest('.position-relative').querySelector('iframe');
                if (iframe) {
                    if (iframe.requestFullscreen) {
                        iframe.requestFullscreen();
                    } else if (iframe.webkitRequestFullscreen) {
                        iframe.webkitRequestFullscreen();
                    } else if (iframe.msRequestFullscreen) {
                        iframe.msRequestFullscreen();
                    }
                }
            });
        });
    }

    // Function to add error message
    function addErrorMessage(message) {
        const timestamp = new Date().toLocaleTimeString();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message assistant-message mb-3 animate__animated animate__fadeIn';

        messageDiv.innerHTML = `
            <div class="message-header d-flex justify-content-between align-items-center mb-1">
                <div class="message-sender">
                    <i class="fas fa-robot me-2"></i> <strong>AI Assistant</strong>
                </div>
                <div class="message-time">
                    ${timestamp}
                </div>
            </div>
            <div class="message-bubble p-3 rounded bg-light">
                <div class="message-content text-danger">
                    <i class="fas fa-exclamation-circle me-2"></i> ${message}
                </div>
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle file selection in dropdown
    const fileInput = document.getElementById('file');
    const selectedFilesChat = document.getElementById('selected-files-chat');

    if (fileInput && selectedFilesChat) {
        fileInput.addEventListener('change', function() {
            // Show selected files in the dropdown
            if (this.files.length > 0) {
                // Create file summary
                const fileCount = this.files.length;
                const fileSummary = document.createElement('div');
                fileSummary.className = 'alert alert-success mb-2 py-2 px-3';
                fileSummary.innerHTML = `
                    <div class="d-flex align-items-center">
                        <i class="fas fa-check-circle me-2"></i>
                        <div>
                            <strong>${fileCount} file${fileCount > 1 ? 's' : ''} selected</strong>
                        </div>
                    </div>
                `;

                // Clear previous selections
                selectedFilesChat.innerHTML = '';
                selectedFilesChat.appendChild(fileSummary);

                // Keep dropdown open after file selection
                const dropdownMenu = document.querySelector('.dropdown-menu');
                if (dropdownMenu) {
                    dropdownMenu.addEventListener('click', function(e) {
                        e.stopPropagation();
                    });
                }
            }
        });
    }

    // Initialize tooltips in the dropdown
    tippy('[data-tippy-content]', {
        allowHTML: true,
        placement: 'bottom',
        theme: 'light-border'
    });
});

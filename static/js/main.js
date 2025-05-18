/**
 * Main JavaScript for Document Research & Theme Identification Chatbot
 *
 * Methodology:
 * -----------
 * 1. User Interface Enhancement:
 *    - Responsive file upload with visual feedback
 *    - Dynamic form handling with loading states
 *    - Interactive knowledge graph controls
 *    - Tabbed interface for results organization
 *
 * 2. Document Processing Visualization:
 *    - Custom animations for document analysis
 *    - Progress indicators for backend operations
 *    - File type detection and appropriate icons
 *    - Size formatting for better readability
 *
 * 3. Chat Interface Management:
 *    - Auto-scrolling message container
 *    - Dynamic textarea resizing
 *    - Message formatting and styling
 *    - Session management and navigation
 *
 * 4. Error Handling:
 *    - Form validation with user feedback
 *    - Graceful fallbacks for animation failures
 *    - Loading state management
 *    - Browser compatibility considerations
 *
 * 5. Knowledge Graph Interaction:
 *    - Fullscreen mode for detailed exploration
 *    - Tab-based content organization
 *    - Responsive iframe handling
 *    - Dynamic content loading
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload preview
    const fileInput = document.getElementById('file');
    const fileName = document.getElementById('file-name');
    const selectedFiles = document.getElementById('selected-files');
    const selectedFilesChat = document.getElementById('selected-files-chat');

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            // Clear previous selections
            if (fileName) fileName.textContent = '';
            if (selectedFiles) selectedFiles.innerHTML = '';
            if (selectedFilesChat) selectedFilesChat.innerHTML = '';

            if (this.files.length > 0) {
                // Show count of files
                const fileCount = this.files.length;

                // Create file summary
                const fileSummary = document.createElement('div');
                fileSummary.className = 'alert alert-success mb-3';
                fileSummary.innerHTML = `
                    <div class="d-flex align-items-center">
                        <i class="fas fa-check-circle me-2 fs-4"></i>
                        <div>
                            <strong>${fileCount} file${fileCount > 1 ? 's' : ''} selected</strong>
                            <div class="small">Ready to analyze</div>
                        </div>
                    </div>
                `;

                if (fileName) {
                    fileName.innerHTML = `<i class="fas fa-files-o me-1"></i> ${fileCount} file${fileCount > 1 ? 's' : ''} selected`;
                }

                // Create file list for display
                const fileList = document.createElement('div');
                fileList.className = 'file-list';

                for (let i = 0; i < this.files.length; i++) {
                    const file = this.files[i];
                    const fileExtension = file.name.split('.').pop().toLowerCase();
                    let iconClass = 'fa-file';
                    let colorClass = 'text-secondary';

                    if (['pdf'].includes(fileExtension)) {
                        iconClass = 'fa-file-pdf';
                        colorClass = 'text-danger';
                    } else if (['png', 'jpg', 'jpeg'].includes(fileExtension)) {
                        iconClass = 'fa-file-image';
                        colorClass = 'text-primary';
                    }

                    const listItem = document.createElement('div');
                    listItem.className = 'file-item p-2 border-bottom';
                    listItem.innerHTML = `
                        <div class="d-flex align-items-center">
                            <i class="fas ${iconClass} me-2 ${colorClass}"></i>
                            <div class="file-details">
                                <div class="file-name">${file.name}</div>
                                <div class="file-size text-muted small">${formatFileSize(file.size)}</div>
                            </div>
                        </div>
                    `;
                    fileList.appendChild(listItem);
                }

                // Add file summary and list to appropriate container
                if (selectedFiles) {
                    selectedFiles.appendChild(fileSummary.cloneNode(true));
                    selectedFiles.appendChild(fileList.cloneNode(true));
                }

                if (selectedFilesChat) {
                    selectedFilesChat.appendChild(fileSummary.cloneNode(true));
                    selectedFilesChat.appendChild(fileList.cloneNode(true));
                }
            }
        });
    }

    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // Form submission loading state
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Check if form has file input
            const hasFileInput = this.querySelector('input[type="file"]');
            const hasQueryInput = this.querySelector('input[name="query"]') || this.querySelector('textarea[name="query"]');
            const fileRequired = hasFileInput && hasFileInput.required;

            // If file input is required but no file is selected, prevent submission
            if (fileRequired && hasFileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a file to upload.');
                return;
            }

            // Validate form
            if (this.checkValidity()) {
                // Get submit button
                const submitBtn = this.querySelector('button[type="submit"]');
                if (submitBtn) {
                    // Show loading state
                    const originalContent = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span> Processing...';
                    submitBtn.disabled = true;

                    // Store original content to restore if there's an error
                    submitBtn.setAttribute('data-original-content', originalContent);
                }

                // Determine if we should show document or query animation
                let animationType = 'query';
                if (hasFileInput && hasFileInput.files.length > 0) {
                    animationType = 'document';
                }

                console.log("Form submitted, showing animation type:", animationType);

                // Directly call the new animation function
                try {
                    console.log("Attempting to show animation of type:", animationType);

                    // Call the new animation function directly from the window object
                    if (typeof window.showNewAnimation === 'function') {
                        console.log("Using showNewAnimation function");
                        window.showNewAnimation(animationType);
                        console.log("Animation function called successfully");
                    } else {
                        console.log("New animation function not available, trying fallback");
                        throw new Error("New animation function not available");
                    }
                } catch (error) {
                    console.error("Error showing animation:", error);

                    // Fallback to basic loading overlay if animation fails
                    let loadingOverlay = document.getElementById('loadingOverlay');

                    // If we're on a page without the overlay, create it
                    if (!loadingOverlay) {
                        loadingOverlay = document.createElement('div');
                        loadingOverlay.id = 'loadingOverlay';
                        loadingOverlay.className = 'loading-overlay';

                        const loadingContent = document.createElement('div');
                        loadingContent.className = 'loading-content';

                        const spinner = document.createElement('div');
                        spinner.className = 'spinner-border text-primary';
                        spinner.setAttribute('role', 'status');
                        spinner.style.width = '3rem';
                        spinner.style.height = '3rem';

                        const spinnerText = document.createElement('span');
                        spinnerText.className = 'visually-hidden';
                        spinnerText.textContent = 'Loading...';

                        const loadingText = document.createElement('p');
                        loadingText.className = 'mt-3 loading-text';

                        // Set loading text based on inputs
                        if (hasFileInput && hasFileInput.files.length > 0) {
                            loadingText.textContent = `Processing ${hasFileInput.files.length} document(s)... This may take a moment.`;
                        } else if (hasQueryInput) {
                            loadingText.textContent = 'Analyzing your question...';
                        } else {
                            loadingText.textContent = 'Processing your request...';
                        }

                        spinner.appendChild(spinnerText);
                        loadingContent.appendChild(spinner);
                        loadingContent.appendChild(loadingText);
                        loadingOverlay.appendChild(loadingContent);

                        document.body.appendChild(loadingOverlay);
                    } else {
                        // Update existing overlay
                        const loadingText = loadingOverlay.querySelector('.loading-text');

                        if (hasFileInput && hasFileInput.files.length > 0) {
                            loadingText.textContent = `Processing ${hasFileInput.files.length} document(s)... This may take a moment.`;
                        } else if (hasQueryInput) {
                            loadingText.textContent = 'Analyzing your question...';
                        } else {
                            loadingText.textContent = 'Processing your request...';
                        }

                        loadingOverlay.classList.remove('d-none');
                    }
                }
            }
        });
    });

    // Reset button confirmation
    const resetBtn = document.getElementById('reset-btn');

    if (resetBtn) {
        resetBtn.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to reset? This will delete all uploaded documents and conversation history.')) {
                e.preventDefault();
            } else {
                resetBtn.innerHTML = '<span class="spinner"></span> Resetting...';
                resetBtn.disabled = true;
            }
        });
    }

    // Knowledge graph interaction enhancements
    const fullscreenBtns = document.querySelectorAll('.fullscreen-btn');

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

    // Auto-scroll chat to bottom
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Auto-resize textarea
    const textareas = document.querySelectorAll('textarea.auto-resize');

    textareas.forEach(textarea => {
        // Initial resize
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';

        // Resize on input
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });

    // Tab functionality for result sections
    const resultTabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    resultTabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            // Resize any iframes in the newly shown tab
            const target = document.querySelector(e.target.getAttribute('data-bs-target'));
            if (target) {
                const iframes = target.querySelectorAll('iframe');
                iframes.forEach(iframe => {
                    // Trigger a resize event
                    const event = new Event('resize');
                    window.dispatchEvent(event);
                });
            }
        });
    });

    // Handle file upload button in chat interface
    const chatFileInput = document.querySelector('.file-upload input[type="file"]');
    const chatFileName = document.querySelector('.file-upload .file-name');

    if (chatFileInput) {
        chatFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                // Update form to indicate file is present
                const form = this.closest('form');
                if (form) {
                    const queryInput = form.querySelector('textarea[name="query"]');
                    if (queryInput) {
                        queryInput.setAttribute('placeholder', 'Ask a question about the new document...');
                    }
                }
            }
        });
    }
});

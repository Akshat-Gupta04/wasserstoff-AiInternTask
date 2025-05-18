/**
 * Document Analysis Animation
 * Shows an animated overlay when analyzing documents or processing queries
 */

// Global variable to track if animation is currently showing
window.animationActive = false;

// Function to show the analysis animation
function showAnalysisAnimation(type = 'query') {
    console.log('Showing analysis animation for: ' + type);

    // If animation is already active, don't create another one
    if (window.animationActive) {
        console.log('Animation already active, not creating another');
        return;
    }

    // Set animation as active
    window.animationActive = true;

    // Check if there's an existing overlay in the HTML
    const existingOverlay = document.getElementById('analysisOverlay');

    if (existingOverlay) {
        console.log('Found existing overlay, using it instead of creating a new one');
        // Show the existing overlay
        existingOverlay.style.display = 'flex';

        // Clear any existing steps
        const processingSteps = existingOverlay.querySelector('.processing-steps');
        if (processingSteps) {
            processingSteps.innerHTML = '';
        }

        // Update the icon based on type
        const docIcon = existingOverlay.querySelector('.doc-icon i');
        if (docIcon) {
            docIcon.className = type === 'document' ? 'fas fa-file-alt' : 'fas fa-brain';
        }

        // Define steps based on type
        let steps = [];
        if (type === 'document') {
            // Steps for document processing
            steps = [
                '<i class="fas fa-file-upload text-primary me-2"></i> Uploading documents...',
                '<i class="fas fa-file-alt text-primary me-2"></i> Extracting text content...',
                '<i class="fas fa-database text-primary me-2"></i> Creating vector embeddings...',
                '<i class="fas fa-brain text-primary me-2"></i> Analyzing document content...',
                '<i class="fas fa-search text-primary me-2"></i> Finding relevant information...'
            ];
        } else {
            // Steps for query processing
            steps = [
                '<i class="fas fa-question-circle text-primary me-2"></i> Processing your question...',
                '<i class="fas fa-search text-primary me-2"></i> Searching for relevant information...',
                '<i class="fas fa-brain text-primary me-2"></i> Analyzing context...',
                '<i class="fas fa-project-diagram text-primary me-2"></i> Generating knowledge graph...',
                '<i class="fas fa-lightbulb text-primary me-2"></i> Formulating response...'
            ];
        }

        // Add steps to the container
        steps.forEach((step, index) => {
            const stepElement = document.createElement('div');
            stepElement.className = 'step';
            stepElement.innerHTML = step;
            processingSteps.appendChild(stepElement);
        });

        // Animate steps appearing one by one
        const stepElements = processingSteps.querySelectorAll('.step');
        let delay = 800; // Initial delay

        stepElements.forEach((step, index) => {
            setTimeout(() => {
                step.classList.add('appear');
            }, delay + (index * 700)); // Add each step with a delay
        });

        // Hide animation after all steps are complete
        const totalDuration = delay + (steps.length * 700) + 1000;
        setTimeout(() => {
            hideAnalysisAnimation();
        }, totalDuration);

        return existingOverlay;
    }

    // If no existing overlay, create a new one
    console.log('No existing overlay found, creating a new one');

    // Remove any existing animation overlays first (just in case)
    if (existingOverlay) {
        existingOverlay.parentNode.removeChild(existingOverlay);
    }

    // Create the analysis overlay
    const analysisOverlay = document.createElement('div');
    analysisOverlay.className = 'analysis-overlay';
    analysisOverlay.id = 'analysisOverlay';

    // Create the content container
    const analysisContent = document.createElement('div');
    analysisContent.className = 'analysis-content';

    // Create animation container
    const animationContainer = document.createElement('div');
    animationContainer.className = 'animation-container';

    // Create document icon
    const docIcon = document.createElement('div');
    docIcon.className = 'doc-icon';
    docIcon.innerHTML = '<i class="fas fa-brain"></i>';

    // Create scan line
    const scanLine = document.createElement('div');
    scanLine.className = 'scan-line';

    // Create processing steps container
    const processingSteps = document.createElement('div');
    processingSteps.className = 'processing-steps';

    // Define steps based on type
    let steps = [];

    if (type === 'document') {
        // Steps for document processing
        steps = [
            { icon: 'fa-file-import', text: 'Importing documents...' },
            { icon: 'fa-font', text: 'Extracting text...' },
            { icon: 'fa-brain', text: 'Processing content...' },
            { icon: 'fa-lightbulb', text: 'Identifying themes...' },
            { icon: 'fa-project-diagram', text: 'Building knowledge graph...' },
            { icon: 'fa-search', text: 'Finding answers...' },
            { icon: 'fa-check-circle', text: 'Analysis complete!' }
        ];
    } else {
        // Steps for query processing
        steps = [
            { icon: 'fa-question-circle', text: 'Processing your question...' },
            { icon: 'fa-search', text: 'Searching documents...' },
            { icon: 'fa-brain', text: 'Analyzing context...' },
            { icon: 'fa-lightbulb', text: 'Finding relevant information...' },
            { icon: 'fa-project-diagram', text: 'Updating knowledge graph...' },
            { icon: 'fa-check-circle', text: 'Analysis complete!' }
        ];
    }

    // Add steps to the container
    steps.forEach((step, index) => {
        const stepElement = document.createElement('div');
        stepElement.className = 'step';
        stepElement.id = `step-${index}`;
        stepElement.innerHTML = `<i class="fas ${step.icon} me-2"></i> ${step.text}`;
        processingSteps.appendChild(stepElement);
    });

    // Add animation container elements
    animationContainer.appendChild(docIcon);
    animationContainer.appendChild(scanLine);

    // Add all to analysis content
    analysisContent.appendChild(animationContainer);
    analysisContent.appendChild(processingSteps);
    analysisOverlay.appendChild(analysisContent);

    // Add to body
    document.body.appendChild(analysisOverlay);

    // Force browser to recognize the new elements
    void analysisOverlay.offsetWidth;

    // Animate steps sequentially
    const stepElements = processingSteps.querySelectorAll('.step');
    let delay = 800; // Initial delay

    stepElements.forEach((step, index) => {
        setTimeout(() => {
            step.classList.add('appear');
        }, delay + (index * 700)); // Add each step with a delay
    });

    // Hide animation after all steps are complete
    const totalDuration = delay + (steps.length * 700) + 1000;
    setTimeout(() => {
        hideAnalysisAnimation();
    }, totalDuration);

    return analysisOverlay;
}

// Function to hide the analysis animation
function hideAnalysisAnimation() {
    const overlay = document.getElementById('analysisOverlay');
    if (overlay) {
        console.log('Hiding analysis animation');

        // For existing overlays in the HTML, just hide them
        if (overlay.hasAttribute('data-permanent')) {
            console.log('Hiding permanent overlay');
            // Just hide it instead of removing it
            overlay.style.display = 'none';
            // Reset animation active flag
            window.animationActive = false;
        } else {
            // For dynamically created overlays, add fade out animation and remove
            console.log('Removing dynamic overlay');
            // Add fade out animation
            overlay.style.animation = 'fadeOut 0.5s ease';

            // Remove after animation completes
            setTimeout(() => {
                if (overlay && overlay.parentNode) {
                    overlay.parentNode.removeChild(overlay);
                }
                // Reset animation active flag
                window.animationActive = false;
            }, 500);
        }
    } else {
        console.log('No overlay found to hide');
        // Reset animation active flag even if overlay not found
        window.animationActive = false;
    }
}

// Add necessary CSS styles directly to the document
function addAnimationStyles() {
    if (!document.querySelector('style#analysis-animation-styles')) {
        const style = document.createElement('style');
        style.id = 'analysis-animation-styles';
        style.textContent = `
            /* Analysis Animation Overlay */
            .analysis-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                animation: fadeIn 0.5s ease;
            }

            .analysis-content {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                max-width: 90%;
                width: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .animation-container {
                position: relative;
                width: 150px;
                height: 150px;
                margin-bottom: 2rem;
            }

            .doc-icon {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 4rem;
                color: #007bff;
                animation: pulse 2s infinite;
            }

            .scan-line {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 5px;
                background-color: #28a745;
                animation: scan 2s infinite;
                box-shadow: 0 0 10px #28a745;
                border-radius: 5px;
            }

            .processing-steps {
                width: 100%;
                text-align: left;
            }

            .step {
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 5px;
                background-color: #f8f9fa;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.3s ease;
            }

            .step.appear {
                opacity: 1;
                transform: translateY(0);
            }

            @keyframes pulse {
                0% {
                    transform: translate(-50%, -50%) scale(1);
                    opacity: 0.7;
                }
                50% {
                    transform: translate(-50%, -50%) scale(1.1);
                    opacity: 1;
                }
                100% {
                    transform: translate(-50%, -50%) scale(1);
                    opacity: 0.7;
                }
            }

            @keyframes scan {
                0% {
                    top: 0;
                }
                50% {
                    top: calc(100% - 5px);
                }
                100% {
                    top: 0;
                }
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }

            @keyframes fadeOut {
                from {
                    opacity: 1;
                }
                to {
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Add the styles when the script loads
addAnimationStyles();

// Make functions available globally
window.showAnalysisAnimation = showAnalysisAnimation;
window.hideAnalysisAnimation = hideAnalysisAnimation;

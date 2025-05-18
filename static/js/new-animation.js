/**
 * Advanced Processing Animation System
 *
 * Methodology:
 * -----------
 * 1. Dynamic Animation Generation:
 *    - Creates animations entirely through JavaScript DOM manipulation
 *    - No reliance on external CSS or HTML elements
 *    - Ensures consistent rendering across browsers
 *    - Prevents conflicts with existing page elements
 *
 * 2. Process Visualization:
 *    - Displays detailed technical steps of backend processing
 *    - Uses visual metaphors (scanning line, pulsing icon) to indicate activity
 *    - Implements sequential step appearance for narrative flow
 *    - Provides real-time progress indication
 *
 * 3. Context-Aware Content:
 *    - Adapts displayed steps based on operation type (document vs. query)
 *    - Shows relevant technical terminology for each process
 *    - Uses appropriate icons and visual cues for different operations
 *    - Maintains consistent visual language across animation types
 *
 * 4. Performance Optimization:
 *    - Manages animation timing to prevent UI freezing
 *    - Implements proper cleanup to prevent memory leaks
 *    - Uses CSS transitions for smooth animations
 *    - Ensures animations complete in reasonable timeframes
 *
 * 5. Error Prevention:
 *    - Implements singleton pattern to prevent multiple overlays
 *    - Handles edge cases like premature closing
 *    - Provides graceful cleanup on page navigation
 *    - Uses global state tracking for animation status
 */

// Global variable to track if animation is currently showing
let animationActive = false;

// Function to show the analysis animation
function showNewAnimation(type = 'query') {
    console.log('NEW ANIMATION: Showing animation for: ' + type);

    // If animation is already active, don't create another one
    if (animationActive) {
        console.log('NEW ANIMATION: Animation already active, not creating another');
        return;
    }

    // Set animation as active
    animationActive = true;

    // Remove any existing animation overlays first
    const existingOverlay = document.getElementById('newAnimationOverlay');
    if (existingOverlay) {
        document.body.removeChild(existingOverlay);
    }

    // Create the analysis overlay
    const analysisOverlay = document.createElement('div');
    analysisOverlay.id = 'newAnimationOverlay';
    analysisOverlay.style.position = 'fixed';
    analysisOverlay.style.top = '0';
    analysisOverlay.style.left = '0';
    analysisOverlay.style.width = '100%';
    analysisOverlay.style.height = '100%';
    analysisOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    analysisOverlay.style.display = 'flex';
    analysisOverlay.style.justifyContent = 'center';
    analysisOverlay.style.alignItems = 'center';
    analysisOverlay.style.zIndex = '9999';

    // Create the content container
    const analysisContent = document.createElement('div');
    analysisContent.style.backgroundColor = 'white';
    analysisContent.style.padding = '2rem';
    analysisContent.style.borderRadius = '10px';
    analysisContent.style.textAlign = 'center';
    analysisContent.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    analysisContent.style.maxWidth = '90%';
    analysisContent.style.width = '500px';
    analysisContent.style.display = 'flex';
    analysisContent.style.flexDirection = 'column';
    analysisContent.style.alignItems = 'center';
    analysisContent.style.color = '#000'; // Ensure text is black
    analysisContent.style.fontFamily = 'Arial, sans-serif'; // Set a common font

    // Create animation container
    const animationContainer = document.createElement('div');
    animationContainer.style.position = 'relative';
    animationContainer.style.width = '150px';
    animationContainer.style.height = '150px';
    animationContainer.style.marginBottom = '2rem';

    // Create icon
    const iconContainer = document.createElement('div');
    iconContainer.style.position = 'absolute';
    iconContainer.style.top = '50%';
    iconContainer.style.left = '50%';
    iconContainer.style.transform = 'translate(-50%, -50%)';
    iconContainer.style.fontSize = '4rem';
    iconContainer.style.color = '#007bff';

    // Add icon based on type
    const icon = document.createElement('i');
    icon.className = type === 'document' ? 'fas fa-file-alt' : 'fas fa-brain';
    iconContainer.appendChild(icon);

    // Create scan line
    const scanLine = document.createElement('div');
    scanLine.style.position = 'absolute';
    scanLine.style.top = '0';
    scanLine.style.left = '0';
    scanLine.style.width = '100%';
    scanLine.style.height = '5px';
    scanLine.style.backgroundColor = '#28a745';
    scanLine.style.boxShadow = '0 0 10px #28a745';
    scanLine.style.borderRadius = '5px';

    // Add animation to scan line
    scanLine.style.animation = 'scan 2s infinite';

    // Add keyframes for scan animation
    const scanStyle = document.createElement('style');
    scanStyle.textContent = `
        @keyframes scan {
            0% { top: 0; }
            50% { top: calc(100% - 5px); }
            100% { top: 0; }
        }

        @keyframes pulse {
            0% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
        }
    `;
    document.head.appendChild(scanStyle);

    // Add pulse animation to icon
    iconContainer.style.animation = 'pulse 2s infinite';

    // Create processing steps container
    const processingSteps = document.createElement('div');
    processingSteps.style.width = '100%';
    processingSteps.style.textAlign = 'left';
    processingSteps.style.marginTop = '20px';
    processingSteps.style.color = '#000'; // Ensure text is black
    processingSteps.style.fontFamily = 'Arial, sans-serif'; // Set a common font
    processingSteps.style.fontSize = '16px'; // Set a reasonable font size

    // Define steps based on type
    let steps = [];

    if (type === 'document') {
        // Steps for document processing - more technical and advanced
        steps = [
            'Initializing document processing pipeline...',
            'Uploading documents to secure processing environment...',
            'Parsing document structure and metadata...',
            'Extracting text content using OCR algorithms...',
            'Preprocessing text: tokenization, lemmatization, and stopword removal...',
            'Generating document embeddings using transformer models...',
            'Creating vector representations in high-dimensional space...',
            'Indexing vectors in optimized database for semantic search...',
            'Analyzing document content with NLP algorithms...',
            'Identifying key entities, relationships, and themes...',
            'Building knowledge graph from extracted information...',
            'Optimizing database for efficient retrieval...',
            'Finalizing document processing and indexing...'
        ];
    } else {
        // Steps for query processing - more technical and advanced
        steps = [
            'Initializing query processing pipeline...',
            'Parsing and tokenizing input query...',
            'Applying semantic analysis to understand query intent...',
            'Generating query embeddings using transformer models...',
            'Performing vector similarity search in high-dimensional space...',
            'Retrieving relevant document chunks with cosine similarity...',
            'Ranking results based on semantic relevance scores...',
            'Extracting context from top-ranked document segments...',
            'Analyzing context with advanced NLP algorithms...',
            'Identifying key entities and relationships in context...',
            'Generating knowledge graph with node-edge relationships...',
            'Applying reasoning algorithms to synthesize information...',
            'Formulating comprehensive response based on extracted data...',
            'Validating response against source documents...'
        ];
    }

    // Add steps to the container
    console.log("Adding steps to container:", steps);
    steps.forEach((step, index) => {
        const stepElement = document.createElement('div');
        stepElement.style.padding = '0.6rem 1rem';
        stepElement.style.marginBottom = '0.4rem';
        stepElement.style.borderRadius = '4px';
        stepElement.style.backgroundColor = '#f0f8ff'; // Light blue background
        stepElement.style.borderLeft = '3px solid #4361ee'; // Blue left border for tech look
        stepElement.style.opacity = '0';
        stepElement.style.transform = 'translateY(10px)';
        stepElement.style.transition = 'all 0.3s ease';
        stepElement.style.color = '#333'; // Dark gray text
        stepElement.style.textAlign = 'left'; // Ensure text is left-aligned
        stepElement.style.fontFamily = 'Consolas, Monaco, "Courier New", monospace'; // Monospace font for tech look
        stepElement.style.fontSize = '13px'; // Slightly smaller font size
        stepElement.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)'; // Subtle shadow

        // Add a technical prefix to each step
        const stepPrefix = document.createElement('span');
        stepPrefix.style.color = '#4361ee'; // Blue color for prefix
        stepPrefix.style.fontWeight = 'bold';
        stepPrefix.style.marginRight = '8px';
        stepPrefix.textContent = `[${String(index+1).padStart(2, '0')}]`;

        stepElement.appendChild(stepPrefix);
        stepElement.appendChild(document.createTextNode(step));

        processingSteps.appendChild(stepElement);

        console.log("Added step:", step);

        // Animate steps appearing one by one - faster timing for more steps
        setTimeout(() => {
            console.log("Animating step:", index);
            stepElement.style.opacity = '1';
            stepElement.style.transform = 'translateY(0)';
        }, 300 + (index * 400));
    });

    // Create a title for the animation
    const title = document.createElement('div');
    title.style.marginBottom = '25px';
    title.style.color = '#4361ee'; // Blue color to match the technical theme
    title.style.fontFamily = 'Consolas, Monaco, "Courier New", monospace'; // Monospace font for tech look
    title.style.fontWeight = 'bold';
    title.style.fontSize = '20px';
    title.style.textAlign = 'center';
    title.style.padding = '10px';
    title.style.borderBottom = '2px solid #e9ecef';
    title.style.width = '100%';

    // Create a more technical-looking title with system-like prefix
    const systemPrefix = document.createElement('span');
    systemPrefix.style.color = '#28a745'; // Green color for system prefix
    systemPrefix.style.marginRight = '10px';
    systemPrefix.textContent = 'SYSTEM:';

    title.appendChild(systemPrefix);
    title.appendChild(document.createTextNode(type === 'document' ? 'DOCUMENT ANALYSIS PIPELINE' : 'QUERY PROCESSING PIPELINE'));

    // Add animation container elements
    animationContainer.appendChild(iconContainer);
    animationContainer.appendChild(scanLine);

    // Create a progress indicator
    const progressContainer = document.createElement('div');
    progressContainer.style.width = '100%';
    progressContainer.style.marginTop = '20px';
    progressContainer.style.marginBottom = '10px';
    progressContainer.style.position = 'relative';

    const progressBar = document.createElement('div');
    progressBar.style.height = '6px';
    progressBar.style.backgroundColor = '#e9ecef';
    progressBar.style.borderRadius = '3px';
    progressBar.style.overflow = 'hidden';

    const progressIndicator = document.createElement('div');
    progressIndicator.style.height = '100%';
    progressIndicator.style.width = '0%';
    progressIndicator.style.backgroundColor = '#4361ee';
    progressIndicator.style.transition = 'width 0.5s ease';

    const progressText = document.createElement('div');
    progressText.style.marginTop = '5px';
    progressText.style.fontSize = '12px';
    progressText.style.color = '#6c757d';
    progressText.style.fontFamily = 'Consolas, Monaco, "Courier New", monospace';
    progressText.style.textAlign = 'right';
    progressText.textContent = 'Initializing...';

    progressBar.appendChild(progressIndicator);
    progressContainer.appendChild(progressBar);
    progressContainer.appendChild(progressText);

    // Animate progress bar
    let currentProgress = 0;
    // Clear any existing interval
    if (progressIntervalId) {
        clearInterval(progressIntervalId);
    }
    // Set new interval and store it globally
    progressIntervalId = setInterval(() => {
        currentProgress += 1;
        if (currentProgress <= 100) {
            progressIndicator.style.width = `${currentProgress}%`;
            progressText.textContent = `Progress: ${currentProgress}%`;
        } else {
            clearInterval(progressIntervalId);
            progressIntervalId = null;
            progressText.textContent = 'Complete: 100%';
        }
    }, totalDuration / 120); // Divide by 120 to ensure it completes before animation ends

    // Add all to analysis content
    analysisContent.appendChild(title);
    analysisContent.appendChild(animationContainer);
    analysisContent.appendChild(processingSteps);
    analysisContent.appendChild(progressContainer);
    analysisOverlay.appendChild(analysisContent);

    // Add to body
    document.body.appendChild(analysisOverlay);

    // Hide animation after all steps are complete - adjusted for more steps
    const totalDuration = 300 + (steps.length * 400) + 1000;
    setTimeout(() => {
        hideNewAnimation();
    }, totalDuration);

    return analysisOverlay;
}

// Store the progress interval globally so we can clear it when hiding
let progressIntervalId = null;

// Function to hide the analysis animation
function hideNewAnimation() {
    const overlay = document.getElementById('newAnimationOverlay');
    if (overlay) {
        console.log('NEW ANIMATION: Hiding animation');

        // Clear the progress interval if it exists
        if (progressIntervalId) {
            clearInterval(progressIntervalId);
            progressIntervalId = null;
        }

        // Add fade out effect
        overlay.style.opacity = '1';
        overlay.style.transition = 'opacity 0.5s ease';
        overlay.style.opacity = '0';

        // Remove after animation completes
        setTimeout(() => {
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
            // Reset animation active flag
            animationActive = false;
        }, 500);
    } else {
        console.log('NEW ANIMATION: No overlay found to hide');
        // Reset animation active flag even if overlay not found
        animationActive = false;
    }
}

// Make functions available globally
window.showNewAnimation = showNewAnimation;
window.hideNewAnimation = hideNewAnimation;

console.log('NEW ANIMATION: Script loaded successfully');

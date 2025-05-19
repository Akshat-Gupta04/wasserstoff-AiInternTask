def process_document_in_memory(file_data, filename, session_id='default'):
    """Process a document with multiple extraction methods for maximum reliability
    without saving to disk. Processes file data directly from memory.

    Methodology:
    -----------
    1. Document Text Extraction:
       - Uses a multi-layered approach with fallback mechanisms
       - Primary: PyMuPDF for direct text extraction from PDFs
       - Secondary: pdfplumber as an alternative PDF text extractor
       - Tertiary: OCR using pytesseract for image-based or scanned documents

    2. Text Processing:
       - Splits text into paragraphs for granular analysis
       - Filters out short or empty paragraphs
       - Normalizes whitespace and formatting
    """
    all_paragraphs = []
    doc_info = []
    
    try:
        # Check if file data exists
        if not file_data or len(file_data) == 0:
            logger.error(f"Empty file data for: {filename}")
            return [], BM25Okapi([[""]]), [{"id": f"{filename}_0_0", "page": 0, "paragraph": 0, "text_preview": "Empty file"}]
        
        # Get file size
        file_size = len(file_data)
        logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")
        
        # Create BytesIO object from file data
        file_stream = io.BytesIO(file_data)
        
        # Process based on file type
        if filename.lower().endswith('.pdf'):
            logger.info(f"Processing PDF file: {filename}")
            
            # Initialize empty list for text sections
            texts = []
            
            # Method 1: Try PyMuPDF first (most reliable)
            if not texts:
                try:
                    logger.info(f"Trying PyMuPDF for {filename}")
                    doc = fitz.open(stream=file_stream, filetype="pdf")
                    logger.info(f"PyMuPDF opened document with {len(doc)} pages")

                    for page_num in range(len(doc)):
                        try:
                            page_text = doc[page_num].get_text()
                            # Add even small amounts of text - we'll filter later if needed
                            if page_text and len(page_text.strip()) > 0:
                                logger.info(f"PyMuPDF extracted {len(page_text)} characters from page {page_num+1}")
                                texts.append((page_num + 1, page_text))
                            else:
                                logger.warning(f"PyMuPDF found no text on page {page_num+1}")
                        except Exception as page_err:
                            logger.error(f"PyMuPDF error on page {page_num+1}: {str(page_err)}")
                except Exception as fitz_err:
                    logger.error(f"PyMuPDF failed: {str(fitz_err)}")
                    # Reset file stream position for next method
                    file_stream.seek(0)
            
            # Method 2: Try pdfplumber if PyMuPDF didn't find text
            if not texts:
                try:
                    logger.info(f"Trying pdfplumber for {filename}")
                    # Reset file stream position
                    file_stream.seek(0)
                    with pdfplumber.open(file_stream) as pdf:
                        logger.info(f"pdfplumber opened document with {len(pdf.pages)} pages")
                        
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and len(page_text.strip()) > 0:
                                    logger.info(f"pdfplumber extracted {len(page_text)} characters from page {page_num+1}")
                                    texts.append((page_num + 1, page_text))
                                else:
                                    logger.warning(f"pdfplumber found no text on page {page_num+1}")
                            except Exception as page_err:
                                logger.error(f"pdfplumber error on page {page_num+1}: {str(page_err)}")
                except Exception as plumber_err:
                    logger.error(f"pdfplumber failed: {str(plumber_err)}")
                    # Reset file stream position for next method
                    file_stream.seek(0)
            
            # Method 3: Last resort - try to extract images and run OCR
            if not texts:
                try:
                    logger.info(f"Trying OCR on PDF images for {filename}")
                    # Reset file stream position
                    file_stream.seek(0)
                    doc = fitz.open(stream=file_stream, filetype="pdf")
                    
                    # Get the best available OCR engine
                    ocr_engine = get_ocr_engine()
                    
                    if ocr_engine != 'none':
                        logger.info(f"Using {ocr_engine} for OCR processing")
                        for page_num in range(len(doc)):
                            try:
                                # Get page as image
                                # Use a reasonable DPI to prevent extremely large images
                                dpi = 150  # Good balance between quality and size
                                pix = doc[page_num].get_pixmap(dpi=dpi)
                                
                                # Check if the pixmap is too large
                                if pix.width > 5000 or pix.height > 5000:
                                    logger.warning(f"Page {page_num+1} too large: {pix.width}x{pix.height}. Using lower DPI.")
                                    # Try again with lower DPI
                                    dpi = 72  # Lower DPI for very large pages
                                    pix = doc[page_num].get_pixmap(dpi=dpi)
                                
                                img_data = pix.tobytes("png")
                                img = Image.open(io.BytesIO(img_data))
                                
                                logger.info(f"Processing page {page_num+1} image: {img.width}x{img.height}")

                                # Run OCR using the best available engine with timeout protection
                                text = perform_ocr(img, engine=ocr_engine, max_retries=1)
                                
                                # Check if we got a timeout message
                                if text and "OCR processing timed out" in text:
                                    logger.warning(f"OCR timed out on page {page_num+1}")
                                    # Add the timeout message as text for this page
                                    texts.append((page_num + 1, f"[OCR processing timed out for page {page_num+1}. The image may be too complex.]"))
                                elif text and len(text.strip()) > 0:
                                    logger.info(f"OCR extracted {len(text)} characters from page {page_num+1} using {ocr_engine}")
                                    texts.append((page_num + 1, text))
                                else:
                                    logger.warning(f"OCR found no text on page {page_num+1} using {ocr_engine}")
                            except Exception as page_err:
                                logger.error(f"OCR error on page {page_num+1}: {str(page_err)}")
                    else:
                        # If no OCR engine is available, add a note about it
                        logger.info(f"Adding note about missing OCR capability for {filename}")
                        ocr_note = f"[OCR processing was skipped for this document because no OCR engine is available. Install Tesseract OCR or add EasyOCR to your environment to process image files.]"
                        texts.append((1, ocr_note))
                except Exception as ocr_err:
                    logger.error(f"PDF OCR failed: {str(ocr_err)}")
            
            # If we still have no text, add a note about it
            if not texts:
                logger.warning(f"No text could be extracted from {filename} using any method")
                texts.append((1, f"[No text could be extracted from this document. The document may be scanned, image-based, or protected.]"))
            
            logger.info(f"Successfully extracted {len(texts)} text sections from {filename}")
            
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.info(f"Processing image file: {filename}")

            # For images, just try OCR directly
            try:
                # Open the image from memory
                image = Image.open(file_stream)
                logger.info(f"Image opened: size={image.size}, mode={image.mode}")
                
                # Check if the image is too large (to prevent memory issues)
                width, height = image.size
                max_dimension = 5000  # Reasonable limit for processing
                
                if width > max_dimension or height > max_dimension:
                    logger.warning(f"Image too large for processing: {width}x{height}. Resizing for safety.")
                    # Resize while maintaining aspect ratio
                    if height > width:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    else:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    
                    try:
                        # Resize using LANCZOS for better quality
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"Image resized to {new_width}x{new_height}")
                    except Exception as resize_err:
                        logger.error(f"Error resizing image: {str(resize_err)}")
                        # If resize fails, add a note and skip OCR
                        texts = [(1, f"[Image too large to process: {width}x{height}. OCR skipped to prevent memory issues.]")]
                        all_paragraphs.append("Image too large to process")
                        doc_info.append({
                            "id": f"{filename}_0_0",
                            "filename": filename,
                            "page": 0,
                            "paragraph": 0,
                            "text_preview": "Image too large to process"
                        })
                        tokenized_paragraphs = [paragraph.split() for paragraph in all_paragraphs]
                        return texts, BM25Okapi(tokenized_paragraphs), doc_info

                # Get the best available OCR engine
                ocr_engine = get_ocr_engine()
                
                if ocr_engine != 'none':
                    logger.info(f"Using {ocr_engine} for OCR processing")
                    
                    # Convert to RGB if needed
                    if image.mode not in ('RGB', 'L'):
                        logger.info(f"Converting image from {image.mode} to RGB")
                        image = image.convert('RGB')

                    # Use the best available OCR engine with timeout protection
                    text = perform_ocr(image, engine=ocr_engine, max_retries=1)
                    
                    # Check if we got a timeout message
                    if text and "OCR processing timed out" in text:
                        logger.warning(f"OCR timed out on image {filename}")
                        # Add the timeout message as text
                        texts = [(1, f"[OCR processing timed out for image {filename}. The image may be too complex.]")]
                    elif text and len(text.strip()) > 0:
                        logger.info(f"OCR extracted {len(text)} characters from image using {ocr_engine}")
                        texts = [(1, text)]
                    else:
                        logger.warning(f"OCR found no text in image using {ocr_engine}")
                        texts = [(1, f"[No text could be extracted from this image.]")]
                else:
                    # If no OCR engine is available, add a note about it
                    logger.info(f"Adding note about missing OCR capability for image {filename}")
                    ocr_note = f"[OCR processing was skipped for this image because no OCR engine is available. Install Tesseract OCR or add EasyOCR to your environment to process image files.]"
                    texts = [(1, ocr_note)]
            except Exception as img_err:
                logger.error(f"Image processing failed: {str(img_err)}")
                texts = [(1, f"[Error processing image: {str(img_err)}]")]
        else:
            logger.error(f"Unsupported file type: {filename}")
            texts = [(1, f"[Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG file.]")]

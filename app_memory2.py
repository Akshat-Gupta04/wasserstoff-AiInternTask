        # Process extracted text sections
        for page_num, text in texts:
            # Split text into paragraphs
            paragraphs = text.split('\n\n')
            
            # Process each paragraph
            for i, para in enumerate(paragraphs):
                # Skip empty paragraphs
                if not para or len(para.strip()) < 10:  # Minimum length to be considered a paragraph
                    continue
                
                # Clean up paragraph
                para = para.strip()
                para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
                
                # Add to list
                all_paragraphs.append(para)
                
                # Add document info
                doc_info.append({
                    "id": f"{filename}_{page_num}_{i}",
                    "filename": filename,
                    "page": page_num,
                    "paragraph": i,
                    "text_preview": para[:100] + "..." if len(para) > 100 else para
                })
        
        # If no paragraphs were extracted, add a placeholder
        if not all_paragraphs:
            all_paragraphs.append("No readable text found in document.")
            doc_info.append({
                "id": f"{filename}_0_0",
                "filename": filename,
                "page": 0,
                "paragraph": 0,
                "text_preview": "No readable text found in document."
            })
        
        # Create BM25 index for the paragraphs
        tokenized_paragraphs = [paragraph.split() for paragraph in all_paragraphs]
        bm25 = BM25Okapi(tokenized_paragraphs)
        
        return texts, bm25, doc_info
    
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")
        return [], BM25Okapi([[""]]), [{"id": f"{filename}_0_0", "page": 0, "paragraph": 0, "text_preview": f"Error: {str(e)}"}]

# Keep the original function for backward compatibility but make it use the in-memory version
def process_document_wrapper(file_path, filename, session_id='default'):
    """Wrapper around process_document_in_memory that reads the file from disk first"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return [], BM25Okapi([[""]]), [{"id": f"{filename}_0_0", "page": 0, "paragraph": 0, "text_preview": "File not found"}]
        
        # Read file into memory
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Process using in-memory function
        return process_document_in_memory(file_data, filename, session_id)
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return [], BM25Okapi([[""]]), [{"id": f"{filename}_0_0", "page": 0, "paragraph": 0, "text_preview": f"Error: {str(e)}"}]

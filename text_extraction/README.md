# Extract text from scanned documents

## Solution 1: regional OCR

Source: https://towardsdatascience.com/extracting-text-from-scanned-pdf-using-pytesseract-open-cv-cd670ee38052

**Data**: NA

**Method**: 
+ Step 1: Converting PDF to Image using pdf2image (dependency: poppler)
+ Step 2: Marking Regions of Image for Information Extraction wtih opencv (cv2.getStructuringElement, cv2.findContours with some preprocessing steps via cv2)
+ Step 3: Applying OCR to the selected region of Image using pytesseract 

**Evaluation**: Look at output

"""
CAD Document Extraction Pipeline
Specialized extraction for engineering CAD drawings (PDF format)
Handles metadata, BOMs, dimensions, and cross-view relationships
"""

import os
import re
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class CADExtractor:
    """Extract structured engineering data from CAD drawings."""
    
    # CAD keywords for detection
    CAD_KEYWORDS = [
        'drawing number', 'model', 'scale', 'revision', 'bom',
        'part number', 'qty', 'description', 'material', 'finish',
        'tolerance', 'dimension', 'view', 'section', 'detail',
        'mm', 'inches', 'unless otherwise specified', '図面番号',
        'title block', 'signature block', 'approval', 'drawing sheet'
    ]
    
    # Engineering industries
    ENGINEERING_INDUSTRIES = {
        'automotive': ['vehicle', 'motor', 'engine', 'chassis', 'wiring', 'harness', 'ecu', 'drive'],
        'manufacturing': ['production', 'assembly', 'tolerance', 'qc', 'iso', 'bom', 'fixture'],
        'engineering': ['voltage', 'resistor', 'schematic', 'pcb', 'torque', 'bearing', 'shaft'],
        'pharma': ['equipment', 'validation', 'batch', 'sterile', 'controlled'],
        'aerospace': ['aircraft', 'avionics', 'flight', 'fuselage', 'landing gear'],
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize CAD extractor."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    def is_cad_document(self, text: str) -> bool:
        """Detect if text contains CAD document indicators."""
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in self.CAD_KEYWORDS if kw in text_lower)
        return keyword_matches >= 3  # At least 3 CAD keywords
    
    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """Extract metadata (drawing number, title, model) from text."""
        metadata = {
            'drawing_number': None,
            'title': None,
            'model': None,
            'scale': None,
            'industry': None,
        }
        
        # Extract drawing number patterns
        drawing_patterns = [
            r'(?:Drawing|DRAWING|図面番号)[\s:]*([A-Z0-9\-\/]+)',
            r'([A-Z]\d{2}[A-Z]\d{4}\d{3}[A-Z]{2})',  # Format like "20230831_99Y_MKR2002100AB"
        ]
        
        for pattern in drawing_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['drawing_number'] = match.group(1)
                break
        
        # Extract model
        model_patterns = [
            r'(?:Model|MODEL|型式)[\s:]*([A-Z0-9\-]+)',
            r'(?:MKR|MKL)\d+[A-Z]{2}',
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['model'] = match.group(0)
                break
        
        # Extract scale
        scale_match = re.search(r'[Ss]cale[\s:]*([0-9:\.]+(?:\s*:?[0-9]*)?)', text)
        if scale_match:
            metadata['scale'] = scale_match.group(1)
        
        # Detect industry from keywords
        for industry, keywords in self.ENGINEERING_INDUSTRIES.items():
            if any(kw in text_lower for kw in keywords):
                metadata['industry'] = industry
                break
        
        return metadata
    
    def extract_bom_from_text(self, text: str) -> List[Dict]:
        """Extract BOM (Bill of Materials) entries from text."""
        bom_entries = []
        
        # Split text into lines and look for BOM patterns
        lines = text.split('\n')
        bom_section = False
        
        for line in lines:
            # Detect BOM section start
            if any(keyword in line.upper() for keyword in ['BOM', 'BILL OF MATERIALS', '部品']):
                bom_section = True
                continue
            
            if not bom_section:
                continue
            
            # Look for rows with numbers and part descriptions
            match = re.match(r'\s*(\d+)\s+([^\d]+?)\s+(\d+)\s*(.*)', line)
            if match:
                entry = {
                    'item': match.group(1),
                    'part_name': match.group(2).strip(),
                    'qty': match.group(3),
                    'notes': match.group(4).strip() if match.group(4) else '',
                }
                bom_entries.append(entry)
        
        return bom_entries
    
    def extract_dimensions_from_text(self, text: str) -> List[Dict]:
        """Extract dimensions from text."""
        dimensions = []
        
        # Pattern for dimensions: number + unit
        dim_pattern = r'(\d+(?:\.\d+)?)\s*(mm|inch|in|″|\'|cm)'
        
        for match in re.finditer(dim_pattern, text):
            dimensions.append({
                'value': float(match.group(1)),
                'unit': match.group(2).replace('″', 'inch').replace('′', 'inch'),
            })
        
        return dimensions
    
    def extract_from_pdf(self, file_path: str, max_pages: int = 3) -> Dict[str, Any]:
        """Extract structured CAD data from PDF."""
        if not HAS_FITZ:
            return {'error': 'PyMuPDF not available', 'extracted_text': ''}
        
        try:
            doc = fitz.open(file_path)
            extracted = {
                'file': file_path,
                'page_count': len(doc),
                'extracted_text': '',
                'metadata': {},
                'bom': [],
                'dimensions': [],
                'is_cad': False,
            }
            
            # Extract text from first N pages
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                extracted['extracted_text'] += f"\n--- PAGE {page_num + 1} ---\n{text}"
            
            # Analyze extracted text
            is_cad = self.is_cad_document(extracted['extracted_text'])
            extracted['is_cad'] = is_cad
            
            if is_cad:
                extracted['metadata'] = self.extract_metadata_from_text(extracted['extracted_text'])
                extracted['bom'] = self.extract_bom_from_text(extracted['extracted_text'])
                extracted['dimensions'] = self.extract_dimensions_from_text(extracted['extracted_text'])
            
            doc.close()
            return extracted
            
        except Exception as e:
            return {
                'error': str(e),
                'file': file_path,
                'is_cad': False,
            }
    
    def call_vision_for_cad(self, file_path: str) -> Optional[Dict]:
        """Use Gemini Vision to analyze CAD drawing image."""
        if not self.model or not HAS_FITZ or not HAS_PIL:
            return None
        
        try:
            doc = fitz.open(file_path)
            first_page = doc[0]
            pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            
            # Encode to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Call Gemini with CAD-specific prompt
            prompt = """
Analyze this engineering CAD drawing and provide:
1. Document type (circuit_diagram, cad_drawing, schematic, etc.)
2. Drawing title/purpose
3. Major components/views visible
4. Industry (automotive, aerospace, manufacturing, electronics, etc.)
5. Drawing format/standard if visible

Return as JSON:
{
    "document_type": "",
    "title": "",
    "views": [],
    "industry": "",
    "standard": "",
    "confidence": 0.9
}
"""
            
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": "image/png",
                    "data": img_b64
                }
            ])
            
            # Parse response
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
        except Exception as e:
            print(f"Vision analysis failed: {e}")
        
        return None
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Complete CAD analysis: text extraction + vision analysis."""
        # Text-based extraction
        text_result = self.extract_from_pdf(file_path)
        
        if not text_result.get('is_cad'):
            return text_result
        
        # Vision-based analysis
        vision_result = self.call_vision_for_cad(file_path)
        
        if vision_result:
            text_result['vision'] = vision_result
            # Use vision data if available
            if not text_result['metadata'].get('industry') and vision_result.get('industry'):
                text_result['metadata']['industry'] = vision_result['industry']
        
        return text_result


def extract_cad_info(file_path: str) -> Dict[str, Any]:
    """Convenience function for CAD extraction."""
    extractor = CADExtractor()
    return extractor.analyze(file_path)

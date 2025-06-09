import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional, Any
import json

class DocumentAPIClient:
    """Client for communicating with the Document Visual Parser API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if the API is running"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception:
            return False
    
    def upload_document(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Upload and process a document"""
        try:
            # Prepare file for upload
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/upload-document",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error uploading document: {str(e)}")
            return None
    
    def analyze_structure(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Analyze document structure only"""
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-structure",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Structure analysis failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error analyzing structure: {str(e)}")
            return None
    
    def analyze_pdf_structure(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Analyze PDF structure only (no image conversion)"""
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-pdf-structure",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"PDF structure analysis failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error analyzing PDF structure: {str(e)}")
            return None
    
    def extract_pdf_pages_into_images(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Extract PDF pages as images"""
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/extract-pdf-pages-into-images",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"PDF extraction failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error extracting PDF pages: {str(e)}")
            return None
    
    def extract_docx_content(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Extract DOCX content"""
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/extract-docx-content",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"DOCX extraction failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error extracting DOCX content: {str(e)}")
            return None
    
    def analyze_docx_with_naive_llm(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Analyze DOCX structure using naive_llm method"""
        try:
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            
            response = self.session.post(
                f"{self.base_url}/api/analyze-docx-with-naive-llm",
                files=files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Naive LLM analysis failed: {response.json().get('detail', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error analyzing DOCX with naive_llm: {str(e)}")
            return None
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        img_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(img_data))


# Singleton instance
api_client = DocumentAPIClient()


def get_api_client() -> DocumentAPIClient:
    """Get the singleton API client instance"""
    return api_client


def set_api_base_url(url: str):
    """Set the API base URL"""
    global api_client
    api_client = DocumentAPIClient(url) 
'use client';

import { useState, useCallback } from 'react';
import { Upload, File, X } from 'lucide-react';
import { motion } from 'framer-motion';

interface DocumentUploaderProps {
  onUpload: (document: any) => void;
}

export default function DocumentUploader({ onUpload }: DocumentUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processFile = async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      // Validate file type
      const allowedTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      ];

      if (!allowedTypes.includes(file.type)) {
        throw new Error('Only PDF and DOCX files are supported');
      }

      // Create FormData for API call
      const formData = new FormData();
      formData.append('file', file);

      // Upload and analyze document
      const response = await fetch('/api/upload-document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload document');
      }

      const result = await response.json();
      
      // Pass document info including formData for chunking
      onUpload({
        ...result,
        formData,
        originalFile: file
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  return (
    <div className="h-full flex flex-col items-center justify-center">
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-red-100 border border-red-300 rounded-lg flex items-center"
        >
          <X className="w-5 h-5 text-red-500 mr-2" />
          <span className="text-red-700">{error}</span>
        </motion.div>
      )}

      <motion.div
        className={`
          relative w-full max-w-md h-64 border-2 border-dashed rounded-lg
          flex flex-col items-center justify-center cursor-pointer
          transition-all duration-200 ease-in-out
          ${isDragOver 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
          }
          ${isUploading ? 'pointer-events-none opacity-50' : ''}
        `}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />

        {isUploading ? (
          <motion.div
            className="flex flex-col items-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            <p className="mt-4 text-gray-600">Processing document...</p>
          </motion.div>
        ) : (
          <motion.div
            className="flex flex-col items-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <Upload className="w-12 h-12 text-gray-400 mb-4" />
            <p className="text-lg font-medium text-gray-700 mb-2">
              Drop your document here
            </p>
            <p className="text-sm text-gray-500 text-center">
              or click to browse
              <br />
              <span className="font-medium">PDF</span> and <span className="font-medium">DOCX</span> files supported
            </p>
          </motion.div>
        )}
      </motion.div>

      <div className="mt-6 flex items-center space-x-4 text-sm text-gray-500">
        <div className="flex items-center">
          <File className="w-4 h-4 mr-1" />
          <span>Max 50MB</span>
        </div>
        <div className="w-1 h-1 bg-gray-300 rounded-full" />
        <span>Secure upload</span>
      </div>
    </div>
  );
} 
'use client';

import { useState } from 'react';
import { FileText, Image as ImageIcon, Play, Loader2, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

interface DocumentViewerProps {
  document: any;
  onStartChunking: () => void;
  isChunking: boolean;
}

export default function DocumentViewer({ document, onStartChunking, isChunking }: DocumentViewerProps) {
  const [currentPage, setCurrentPage] = useState(0);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeIcon = (fileType: string) => {
    if (fileType.includes('pdf')) {
      return <FileText className="w-6 h-6 text-red-500" />;
    } else if (fileType.includes('document')) {
      return <FileText className="w-6 h-6 text-blue-500" />;
    }
    return <FileText className="w-6 h-6 text-gray-500" />;
  };

  return (
    <div className="h-full flex flex-col">
      {/* Document Info Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-50 rounded-lg p-4 mb-4"
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            {getFileTypeIcon(document.file_type)}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 truncate max-w-64">
                {document.filename}
              </h3>
              <p className="text-sm text-gray-600">
                {formatFileSize(document.size)}
                {document.total_pages && ` • ${document.total_pages} pages`}
              </p>
            </div>
          </div>
          
          <motion.button
            onClick={onStartChunking}
            disabled={isChunking}
            className={`
              flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all
              ${isChunking 
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                : 'bg-blue-500 text-white hover:bg-blue-600 hover:shadow-lg'
              }
            `}
            whileHover={!isChunking ? { scale: 1.05 } : {}}
            whileTap={!isChunking ? { scale: 0.95 } : {}}
          >
            {isChunking ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Chunking...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Chunking</span>
              </>
            )}
          </motion.button>
        </div>
      </motion.div>

      {/* Document Content */}
      <div className="flex-1 overflow-hidden">
        {document.pages && document.pages.length > 0 ? (
          <div className="h-full flex flex-col">
            {/* Page Navigation */}
            {document.pages.length > 1 && (
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                    disabled={currentPage === 0}
                    className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                  >
                    <RefreshCw className="w-4 h-4 rotate-180" />
                  </button>
                  <span className="text-sm text-gray-600">
                    Page {currentPage + 1} of {document.pages.length}
                  </span>
                  <button
                    onClick={() => setCurrentPage(Math.min(document.pages.length - 1, currentPage + 1))}
                    disabled={currentPage === document.pages.length - 1}
                    className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Page Display */}
            <div className="flex-1 overflow-auto bg-gray-100 rounded-lg p-4">
              <motion.div
                key={currentPage}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
                className="flex justify-center"
              >
                <img
                  src={`data:image/png;base64,${document.pages[currentPage].image_data}`}
                  alt={`Page ${currentPage + 1}`}
                  className="max-w-full h-auto shadow-lg rounded border"
                />
              </motion.div>
            </div>
          </div>
        ) : (
          // Document Structure Preview
          <div className="h-full overflow-auto bg-gray-50 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              Document Structure
            </h4>
            
            {document.structure && document.structure.length > 0 ? (
              <div className="space-y-2">
                {document.structure.slice(0, 10).map((item: any, index: number) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`
                      p-3 rounded-lg border-l-4 bg-white
                      ${item.level === 1 ? 'border-blue-500' : 
                        item.level === 2 ? 'border-green-500' : 
                        'border-gray-300'}
                    `}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-800 truncate">
                        {item.title || item.text || 'Untitled Section'}
                      </span>
                      <span className="text-xs text-gray-500 ml-2">
                        Level {item.level || 1}
                      </span>
                    </div>
                  </motion.div>
                ))}
                {document.structure.length > 10 && (
                  <div className="text-center text-gray-500 text-sm">
                    ... and {document.structure.length - 10} more sections
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                <ImageIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Document structure will appear here</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
} 
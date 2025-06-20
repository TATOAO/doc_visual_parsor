'use client';

import { useState } from 'react';
import DocumentUploader from '@/components/DocumentUploader';
import DocumentViewer from '@/components/DocumentViewer';
import ChunkingVisualizer from '@/components/ChunkingVisualizer';
import { ChunkingData, ChunkData } from '@/types/chunking';

export default function Home() {
  const [uploadedDocument, setUploadedDocument] = useState<any>(null);
  const [chunkingData, setChunkingData] = useState<ChunkingData | null>(null);
  const [isChunking, setIsChunking] = useState(false);

  const handleDocumentUpload = (document: any) => {
    setUploadedDocument(document);
    setChunkingData(null);
  };

  // Helper function to convert Section to ChunkData format
  const convertSectionToChunkData = (section: any, index: number): ChunkData => {
    return {
      id: section.section_hash || `section_${index}_${Date.now()}`,
      title: section.title || `Section ${index + 1}`,
      content: section.content || '',
      level: section.level || 1,
      parent_id: undefined, // Will be set during tree processing
      children: section.sub_sections ? section.sub_sections.map((subSection: any, subIndex: number) => 
        convertSectionToChunkData(subSection, subIndex)
      ) : [],
      timestamp: Date.now()
    };
  };

  // Helper function to flatten section tree into array of chunks
  const flattenSections = (sections: ChunkData[], parentId?: string): ChunkData[] => {
    const flattened: ChunkData[] = [];
    
    for (const section of sections) {
      const chunk: ChunkData = {
        ...section,
        parent_id: parentId
      };
      
      flattened.push(chunk);
      
      if (section.children && section.children.length > 0) {
        const childChunks = flattenSections(section.children, section.id);
        flattened.push(...childChunks);
      }
    }
    
    return flattened;
  };

  const handleStartChunking = async () => {
    if (!uploadedDocument) return;
    
    setIsChunking(true);
    setChunkingData({ chunks: [], status: 'processing', progress: 0 });
    
    try {
      // Call the backend API directly
      const response = await fetch('http://localhost:8000/api/chunk-document-sse', {
        method: 'POST',
        body: uploadedDocument.formData,
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.replace('data: ', '').trim();
            
            if (data === '[DONE]') {
              setIsChunking(false);
              setChunkingData(prev => prev ? { ...prev, status: 'completed' } : null);
              return;
            }
            
            try {
              const parsedData = JSON.parse(data);
              
              if (parsedData.success === true) {
                // End event
                setIsChunking(false);
                setChunkingData(prev => prev ? { ...prev, status: 'completed' } : null);
                return;
              }
              
              // Convert Section to ChunkData and flatten - REPLACE instead of accumulate
              const chunkData = convertSectionToChunkData(parsedData, 0);
              const flattenedChunks = flattenSections([chunkData]);
              
              // Replace chunks with latest result instead of accumulating
              setChunkingData(prev => ({
                chunks: flattenedChunks, // Replace entirely with latest data
                status: 'processing',
                progress: Math.min(flattenedChunks.length * 10, 100) // Base progress on current chunk count
              }));
              
            } catch (e) {
              console.error('Error parsing chunk data:', e, 'Data:', data);
            }
          } else if (line.startsWith('event: ')) {
            // Handle event type if needed
            const eventType = line.replace('event: ', '').trim();
            console.log('Event type:', eventType);
          }
        }
      }
    } catch (error) {
      console.error('Error chunking document:', error);
      setIsChunking(false);
      setChunkingData(prev => prev ? { ...prev, status: 'error' } : null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Document Chunking Visualizer
          </h1>
          <p className="text-gray-600 text-lg">
            Upload a document and watch the AI chunking process in real-time
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[calc(100vh-200px)]">
          {/* Left Panel - Document Display */}
          <div className="bg-white rounded-lg shadow-lg p-6 overflow-hidden">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Document
            </h2>
            
            {!uploadedDocument ? (
              <DocumentUploader onUpload={handleDocumentUpload} />
            ) : (
              <DocumentViewer 
                document={uploadedDocument} 
                onStartChunking={handleStartChunking}
                isChunking={isChunking}
              />
            )}
          </div>

          {/* Right Panel - Chunking Process */}
          <div className="bg-white rounded-lg shadow-lg p-6 overflow-hidden">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Chunking Process
            </h2>
            
            <ChunkingVisualizer 
              chunkingData={chunkingData}
              isChunking={isChunking}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

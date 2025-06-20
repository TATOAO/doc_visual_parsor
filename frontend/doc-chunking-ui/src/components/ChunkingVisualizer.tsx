'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Layers, Clock, CheckCircle2, Circle, Zap } from 'lucide-react';
import { ChunkingData } from '@/types/chunking';

interface ChunkingVisualizerProps {
  chunkingData: ChunkingData | null;
  isChunking: boolean;
}

interface ChunkDisplay {
  id: string;
  title: string;
  level: number;
  timestamp: number;
  status: 'appearing' | 'visible' | 'processing';
}

export default function ChunkingVisualizer({ chunkingData, isChunking }: ChunkingVisualizerProps) {
  const [displayedChunks, setDisplayedChunks] = useState<ChunkDisplay[]>([]);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (chunkingData?.chunks) {
      const newChunks = chunkingData.chunks.map(chunk => ({
        id: chunk.id,
        title: chunk.title,
        level: chunk.level,
        timestamp: chunk.timestamp,
        status: 'appearing' as const
      }));
      
      setDisplayedChunks(newChunks);
      
      // Update progress
      if (chunkingData.chunks.length > 0) {
        setProgress((chunkingData.chunks.length / 20) * 100); // Assuming max 20 chunks
      }
    }
  }, [chunkingData]);

  const getIndentLevel = (level: number) => {
    return Math.max(0, (level - 1) * 24);
  };

  const getLevelColor = (level: number) => {
    const colors = [
      'text-blue-600 border-blue-200',
      'text-green-600 border-green-200',
      'text-purple-600 border-purple-200',
      'text-orange-600 border-orange-200',
      'text-pink-600 border-pink-200'
    ];
    return colors[Math.min(level - 1, colors.length - 1)] || colors[0];
  };

  const getChunkIcon = (level: number) => {
    switch (level) {
      case 1:
        return <FileText className="w-4 h-4" />;
      case 2:
        return <Layers className="w-4 h-4" />;
      default:
        return <Circle className="w-3 h-3" />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Status Header */}
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            {isChunking ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <Zap className="w-5 h-5 text-blue-500" />
                </motion.div>
                <span className="text-lg font-semibold text-gray-800">Processing...</span>
              </>
            ) : displayedChunks.length > 0 ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-green-500" />
                <span className="text-lg font-semibold text-gray-800">Completed</span>
              </>
            ) : (
              <>
                <Clock className="w-5 h-5 text-gray-400" />
                <span className="text-lg font-semibold text-gray-600">Ready to start</span>
              </>
            )}
          </div>
          
          {displayedChunks.length > 0 && (
            <div className="text-sm text-gray-600">
              {displayedChunks.length} chunks extracted
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {(isChunking || displayedChunks.length > 0) && (
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-blue-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(progress, 100)}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        )}
      </div>

      {/* Chunks Display */}
      <div className="flex-1 overflow-auto">
        {!isChunking && displayedChunks.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-400">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <Layers className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">Ready for chunking</p>
              <p className="text-sm">
                Upload a document and click "Start Chunking" to see the AI break it down into sections
              </p>
            </motion.div>
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence>
              {displayedChunks.map((chunk, index) => (
                <motion.div
                  key={chunk.id}
                  initial={{ opacity: 0, x: -50, scale: 0.9 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{
                    duration: 0.5,
                    delay: isChunking ? index * 0.2 : 0,
                    type: "spring",
                    stiffness: 100
                  }}
                  className="relative"
                  style={{ marginLeft: `${getIndentLevel(chunk.level)}px` }}
                >
                  {/* Connection Line for nested items */}
                  {chunk.level > 1 && (
                    <div
                      className="absolute left-0 top-0 w-px bg-gray-200"
                      style={{
                        height: '100%',
                        left: `${-12}px`
                      }}
                    />
                  )}

                  <div className={`
                    p-4 rounded-lg border-l-4 bg-white shadow-sm hover:shadow-md
                    transition-all duration-200 ${getLevelColor(chunk.level)}
                  `}>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getChunkIcon(chunk.level)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <motion.h4
                            className="text-sm font-semibold text-gray-800 truncate"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                          >
                            {chunk.title || `Section ${index + 1}`}
                          </motion.h4>
                          
                          <span className="text-xs text-gray-500 ml-2">
                            Level {chunk.level}
                          </span>
                        </div>
                        
                        {isChunking && (
                          <motion.div
                            className="flex items-center space-x-1 text-xs text-gray-500"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.5 }}
                          >
                            <div className="w-1 h-1 bg-green-500 rounded-full animate-pulse" />
                            <span>Processing...</span>
                          </motion.div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading indicator for new chunks */}
            {isChunking && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center justify-center p-4 text-gray-500"
              >
                <motion.div
                  className="flex space-x-1"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  {[0, 1, 2].map((dot) => (
                    <motion.div
                      key={dot}
                      className="w-2 h-2 bg-blue-500 rounded-full"
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.7, 1, 0.7]
                      }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        delay: dot * 0.2
                      }}
                    />
                  ))}
                </motion.div>
                <span className="ml-3 text-sm">Looking for more chunks...</span>
              </motion.div>
            )}
          </div>
        )}
      </div>

      {/* Summary Footer */}
      {displayedChunks.length > 0 && !isChunking && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
              <span className="text-green-800 font-medium">
                Chunking completed successfully!
              </span>
            </div>
            <div className="text-sm text-green-600">
              {displayedChunks.length} sections identified
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
} 
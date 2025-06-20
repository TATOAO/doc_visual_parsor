export interface ChunkData {
  id: string;
  title: string;
  content: string;
  level: number;
  parent_id?: string;
  children?: ChunkData[];
  timestamp: number;
}

export interface ChunkingData {
  chunks: ChunkData[];
  status: 'processing' | 'completed' | 'error';
  progress?: number;
}

export interface DocumentInfo {
  filename: string;
  file_type: string;
  size: number;
  total_pages?: number;
  structure?: any;
  pages?: Array<{
    page_number: number;
    image_data: string;
  }>;
  formData: FormData;
}

export interface ChunkingProgress {
  current_chunk: number;
  total_chunks: number;
  current_title: string;
  status: 'starting' | 'processing' | 'completed';
} 
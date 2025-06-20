# Document Chunking Visualizer Frontend

A modern React/Next.js frontend for visualizing the AI-powered document chunking process. Upload documents and watch in real-time as the AI breaks them down into structured sections.

## Features

- 🔄 **Real-time Chunking**: Watch the AI process your document with streaming animations
- 📄 **Multi-format Support**: Supports PDF and DOCX files
- 🎨 **Modern UI**: Beautiful, responsive interface built with Tailwind CSS
- ⚡ **Fast Performance**: Optimized with Next.js and Framer Motion animations
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Python backend server (see main README)

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend/doc-chunking-ui
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Running with Backend

For the complete experience, you need both the backend API and frontend running:

```bash
# From the root directory
python run_frontend.py
```

This will start:
- Backend API server at `http://localhost:8000`
- Frontend at `http://localhost:3000`

## How to Use

1. **Upload Document**: Drag and drop or click to select a PDF or DOCX file
2. **View Document**: See your document's structure and pages (for PDFs)
3. **Start Chunking**: Click the "Start Chunking" button to begin the AI processing
4. **Watch Animation**: See chunks appear in real-time with smooth animations
5. **Explore Results**: Browse the hierarchical structure of your document

## API Integration

The frontend connects to the backend API with these main endpoints:

- `POST /api/upload-document` - Upload and analyze document
- `POST /api/chunk-document-sse` - Stream chunking results via Server-Sent Events

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **UI**: React 19 with TypeScript
- **Styling**: Tailwind CSS 4
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **HTTP Client**: Native Fetch API

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Main application page
│   └── layout.tsx            # Root layout
├── components/
│   ├── DocumentUploader.tsx  # File upload component
│   ├── DocumentViewer.tsx    # Document display component
│   └── ChunkingVisualizer.tsx # Real-time chunking animation
└── types/
    └── chunking.ts           # TypeScript interfaces
```

## Development

### Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Configuration

- `next.config.ts` - Next.js configuration with API proxying
- `tailwind.config.ts` - Tailwind CSS configuration
- `tsconfig.json` - TypeScript configuration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Upload,
  FileVideo,
  AlertCircle,
  CheckCircle2,
  Download,
  Terminal,
  Activity,
  ArrowRight,
  ChevronDown,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ProcessingStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  output_file?: string;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"inference" | "documentation">(
    "inference"
  );
  const [viewState, setViewState] = useState<
    "idle" | "uploading" | "processing" | "complete" | "error"
  >("idle");
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [readmeContent, setReadmeContent] = useState<string>("");
  const [readmeLoading, setReadmeLoading] = useState<boolean>(true);
  const [pipelineType, setPipelineType] = useState<"original" | "saliency">(
    "original"
  );

  const fileInputRef = useRef<HTMLInputElement>(null);
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  // Load README content when documentation tab is active
  useEffect(() => {
    if (activeTab === "documentation") {
      loadReadme();
    }
  }, [activeTab]);

  const loadReadme = async () => {
    setReadmeLoading(true);
    try {
      const response = await fetch("/api/readme");
      if (response.ok) {
        const data = await response.json();
        setReadmeContent(data.content);
      } else {
        setReadmeContent("# Error\n\nFailed to load documentation.");
      }
    } catch (err) {
      setReadmeContent("# Error\n\nFailed to load documentation.");
    } finally {
      setReadmeLoading(false);
    }
  };

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  const handleFileSelect = (selectedFile: File) => {
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
      setError(null);
      addLog(
        `File selected: ${selectedFile.name} (${(
          selectedFile.size /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
    } else {
      setError("Please upload a valid video file (MP4, AVI, etc.)");
      addLog("Error: Invalid file type selected.");
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const startUpload = async () => {
    if (!file) return;

    setViewState("uploading");
    setStatusMessage("Uploading video to processing unit...");
    addLog(
      `Initiating upload sequence... (Pipeline: ${
        pipelineType === "original" ? "Original D2-City" : "Saliency-Enhanced"
      })`
    );

    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf_threshold", "0.5");
    formData.append("pipeline_type", pipelineType);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const data = response.data;
      setJobId(data.job_id);
      addLog(`Upload complete. Job ID: ${data.job_id}`);

      setViewState("processing");
      startPolling(data.job_id);
    } catch (err: any) {
      console.error(err);
      const errorMsg =
        err.response?.data?.detail || err.message || "Connection failed";
      setError(errorMsg);
      setViewState("error");
      addLog(`Critical Error: ${errorMsg}`);
    }
  };

  const startPolling = (id: string) => {
    addLog("Starting inference pipeline...");
    statusIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/status/${id}`);
        const data = response.data as ProcessingStatus;

        if (data.progress) {
          setProgress(Math.round(data.progress * 100));
        }
        setStatusMessage(data.message || "Processing frames...");

        if (
          data.status === "completed" ||
          data.status === "success" ||
          data.progress === 1.0
        ) {
          stopPolling();
          handleCompletion(id, data.output_file);
        } else if (data.status === "failed" || data.status === "error") {
          stopPolling();
          throw new Error(
            data.message || "Server reported processing failure."
          );
        }
      } catch (err: any) {
        stopPolling();
        const errorMsg =
          err.response?.data?.detail || err.message || "Status check failed";
        setError(errorMsg);
        setViewState("error");
        addLog(`Error during processing: ${errorMsg}`);
      }
    }, 2000);
  };

  const stopPolling = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
    }
  };

  const handleCompletion = (id: string, filename?: string) => {
    setViewState("complete");
    setStatusMessage("Inference complete.");
    addLog("Processing finished. Retrieving output.");
    setOutputUrl(`${API_URL}/download/${id}`);
  };

  const resetApp = () => {
    setViewState("idle");
    setFile(null);
    setJobId(null);
    setProgress(0);
    setLogs([]);
    setError(null);
    setOutputUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-white selection:text-black flex flex-col items-center px-6">
      <div className="w-full max-w-4xl flex flex-col h-screen">
        {/* Header */}
        <header className="w-full py-6 border-b border-white/10 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold tracking-tight">
              RT-DETR -{" "}
              <span className="font-light opacity-60">
                Driver Assistance System
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono opacity-50">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            SYSTEM ONLINE
          </div>
        </header>

        {/* Tabs */}
        <div className="w-full">
          <div className="flex gap-8">
            <button
              onClick={() => setActiveTab("inference")}
              className={`px-4 py-3 text-sm font-medium transition-colors relative ${
                activeTab === "inference"
                  ? "text-white"
                  : "text-neutral-500 hover:text-neutral-300"
              }`}
            >
              Inference
              {activeTab === "inference" && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white"></div>
              )}
            </button>
            <button
              onClick={() => setActiveTab("documentation")}
              className={`px-4 py-3 text-sm font-medium transition-colors relative ${
                activeTab === "documentation"
                  ? "text-white"
                  : "text-neutral-500 hover:text-neutral-300"
              }`}
            >
              Documentation
              {activeTab === "documentation" && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white"></div>
              )}
            </button>
          </div>
        </div>

        <main
          className={`flex-1 flex flex-col ${
            activeTab === "inference" ? "justify-center" : ""
          } py-10 gap-12 overflow-auto`}
        >
          {/* INFERENCE TAB */}
          {activeTab === "inference" && (
            <>
              {/* STATE: IDLE / FILE SELECTION */}
              {viewState === "idle" && (
                <div className="flex flex-col items-center animate-in fade-in duration-700 gap-6">
                  {/* Pipeline Type Selector */}
                  <div className="w-full max-w-2xl">
                    <label className="block text-sm font-medium text-neutral-400 mb-2">
                      Pipeline Type
                    </label>
                    <div className="relative">
                      <select
                        value={pipelineType}
                        onChange={(e) =>
                          setPipelineType(
                            e.target.value as "original" | "saliency"
                          )
                        }
                        className="w-full bg-neutral-900 border border-white/10 text-white px-4 py-3 rounded-sm appearance-none cursor-pointer hover:border-white/20 transition-colors focus:outline-none focus:border-white/40"
                        aria-label="Select pipeline type"
                      >
                        <option value="original">
                          Original D2-City (with preprocessing)
                        </option>
                        <option value="saliency">
                          Saliency-Enhanced (pre-processed frames)
                        </option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400 pointer-events-none" />
                    </div>
                    <p className="text-xs text-neutral-500 mt-2">
                      {pipelineType === "original"
                        ? "Uses original D2-City dataset with data loading and preprocessing"
                        : "Uses pre-processed saliency-enhanced frames (no preprocessing needed)"}
                    </p>
                  </div>

                  <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={onDragOver}
                    onDrop={onDrop}
                    className="group w-full max-w-2xl aspect-video border border-dashed border-white/20 hover:border-white hover:bg-white/5 transition-all duration-300 rounded-lg flex flex-col items-center justify-center cursor-pointer relative overflow-hidden"
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleInputChange}
                      className="hidden"
                      accept="video/*"
                      aria-label="Upload dashcam video file"
                    />

                    {!file ? (
                      <>
                        <div className="w-16 h-16 mb-6 rounded-full border border-white/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                          <Upload className="w-6 h-6 text-white/70" />
                        </div>
                        <h3 className="text-lg font-medium">
                          Upload Dashcam Footage
                        </h3>
                        <p className="text-sm text-neutral-500 mt-2">
                          Drag & drop or click to browse
                        </p>
                        <div className="absolute bottom-4 text-[10px] text-neutral-700 font-mono">
                          Accepts MP4, AVI • Max 500MB
                        </div>
                      </>
                    ) : (
                      <div className="flex flex-col items-center z-10">
                        <FileVideo className="w-12 h-12 mb-4 text-white" />
                        <span className="text-lg font-mono">{file.name}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startUpload();
                          }}
                          className="mt-6 bg-white text-black px-8 py-3 rounded-sm font-semibold hover:bg-neutral-200 transition-colors flex items-center gap-2"
                        >
                          Initialize Processing{" "}
                          <ArrowRight className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setFile(null);
                          }}
                          className="mt-4 text-xs text-neutral-500 hover:text-white underline"
                        >
                          Change File
                        </button>
                      </div>
                    )}
                  </div>
                  {error && (
                    <div className="mt-4 flex items-center gap-2 text-red-500 text-sm font-mono border border-red-900/30 bg-red-900/10 px-4 py-2 rounded">
                      <AlertCircle className="w-4 h-4" /> {error}
                    </div>
                  )}
                </div>
              )}

              {/* STATE: PROCESSING / UPLOADING */}
              {(viewState === "uploading" || viewState === "processing") && (
                <div className="w-full max-w-2xl mx-auto flex flex-col gap-8 animate-in fade-in zoom-in-95 duration-500">
                  <div className="border border-white/10 bg-neutral-950 p-8 rounded-lg relative overflow-hidden">
                    <div className="flex justify-between items-end mb-4">
                      <div>
                        <h2 className="text-2xl font-light mb-1">
                          {viewState === "uploading"
                            ? "Uploading Data"
                            : "Analyzing Scene"}
                        </h2>
                        <p className="text-sm text-neutral-500 font-mono uppercase tracking-wider">
                          {statusMessage}
                        </p>
                      </div>
                      <span className="text-4xl font-bold font-mono">
                        {progress}%
                      </span>
                    </div>

                    <div className="w-full h-1 bg-neutral-900 mt-4 overflow-hidden">
                      <div
                        className="h-full bg-white transition-all duration-500 ease-out"
                        style={{ width: `${progress}%` }}
                      />
                    </div>

                    <div className="absolute top-0 right-0 p-4 opacity-20">
                      <Activity className="w-12 h-12 text-white animate-pulse" />
                    </div>
                  </div>

                  {/* Terminal Log Output */}
                  <div className="bg-black border border-white/10 p-4 rounded-sm h-48 overflow-y-auto font-mono text-xs text-neutral-400 custom-scrollbar">
                    <div className="flex items-center gap-2 text-white mb-2 pb-2 border-b border-white/10 sticky top-0 bg-black">
                      <Terminal className="w-3 h-3" /> System Log
                    </div>
                    {logs.map((log, i) => (
                      <div key={i} className="mb-1">
                        {log}
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </div>
                </div>
              )}

              {/* STATE: COMPLETE */}
              {viewState === "complete" && outputUrl && (
                <div className="flex flex-col lg:flex-row gap-8 h-full animate-in slide-in-from-bottom-4 duration-700">
                  <div className="flex-1 flex flex-col bg-neutral-900/20 border border-white/10 rounded-lg overflow-hidden">
                    <div className="bg-neutral-900/50 p-3 border-b border-white/10 flex justify-between items-center">
                      <span className="text-xs font-mono text-neutral-400">
                        OUTPUT_RENDER_FINAL.mp4
                      </span>
                      <span className="flex items-center gap-1 text-[10px] bg-green-900/30 text-green-400 px-2 py-1 rounded border border-green-900/50">
                        <CheckCircle2 className="w-3 h-3" /> PROCESSED
                      </span>
                    </div>

                    <div className="flex-1 bg-black relative flex items-center justify-center group">
                      <video
                        src={outputUrl}
                        controls
                        className="max-h-[60vh] w-full object-contain"
                      />
                    </div>

                    <div className="p-4 border-t border-white/10 flex justify-between items-center bg-black">
                      <button
                        onClick={resetApp}
                        className="text-sm text-neutral-500 hover:text-white transition-colors"
                      >
                        Process New Video
                      </button>
                      <a
                        href={outputUrl}
                        download={`rtdetr_output_${jobId}.mp4`}
                        className="bg-white text-black px-6 py-2 rounded-sm font-semibold hover:bg-neutral-200 transition-colors flex items-center gap-2 text-sm"
                      >
                        <Download className="w-4 h-4" /> Download Result
                      </a>
                    </div>
                  </div>
                </div>
              )}

              {/* STATE: ERROR */}
              {viewState === "error" && (
                <div className="flex flex-col items-center justify-center animate-in zoom-in-95">
                  <div className="bg-neutral-900/50 border border-red-900/50 p-8 rounded-lg max-w-md text-center">
                    <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <h2 className="text-xl font-bold text-white mb-2">
                      Processing Failed
                    </h2>
                    <p className="text-neutral-400 text-sm mb-6">{error}</p>
                    <div className="bg-black p-3 rounded border border-white/10 text-xs font-mono text-left mb-6 text-red-400">
                      Status: 500
                      <br />
                      Job ID: {jobId || "N/A"}
                    </div>
                    <button
                      onClick={resetApp}
                      className="bg-white text-black px-6 py-2 rounded-sm font-semibold hover:bg-neutral-200 transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              )}
            </>
          )}

          {/* DOCUMENTATION TAB */}
          {activeTab === "documentation" && (
            <div className="w-full max-w-4xl mx-auto px-4 py-4">
              {readmeLoading ? (
                <div className="flex items-center justify-center py-20">
                  <div className="text-neutral-500 font-mono">
                    Loading documentation...
                  </div>
                </div>
              ) : (
                <div className="prose prose-invert prose-lg max-w-none text-white">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ node, ...props }) => (
                        <h1
                          className="text-3xl font-bold mb-4 text-white"
                          {...props}
                        />
                      ),
                      h2: ({ node, ...props }) => (
                        <h2
                          className="text-2xl font-bold mt-8 mb-4 text-white"
                          {...props}
                        />
                      ),
                      h3: ({ node, ...props }) => (
                        <h3
                          className="text-xl font-bold mt-6 mb-3 text-white"
                          {...props}
                        />
                      ),
                      p: ({ node, ...props }) => (
                        <p
                          className="mb-4 text-neutral-300 leading-relaxed"
                          {...props}
                        />
                      ),
                      ul: ({ node, ...props }) => (
                        <ul
                          className="list-disc list-inside mb-4 text-neutral-300 space-y-2"
                          {...props}
                        />
                      ),
                      ol: ({ node, ...props }) => (
                        <ol
                          className="list-decimal list-inside mb-4 text-neutral-300 space-y-2"
                          {...props}
                        />
                      ),
                      li: ({ node, ...props }) => (
                        <li className="text-neutral-300" {...props} />
                      ),
                      code: ({ node, inline, ...props }: any) => {
                        if (inline) {
                          return (
                            <code
                              className="bg-neutral-900 text-white px-1.5 py-0.5 rounded text-sm font-mono"
                              {...props}
                            />
                          );
                        }
                        return (
                          <code
                            className="block bg-neutral-900 text-white p-4 rounded mb-4 overflow-x-auto text-sm font-mono"
                            {...props}
                          />
                        );
                      },
                      pre: ({ node, ...props }) => (
                        <pre
                          className="bg-neutral-900 p-4 rounded mb-4 overflow-x-auto"
                          {...props}
                        />
                      ),
                      blockquote: ({ node, ...props }) => (
                        <blockquote
                          className="border-l-4 border-neutral-700 pl-4 italic text-neutral-400 mb-4"
                          {...props}
                        />
                      ),
                      a: ({ node, ...props }) => (
                        <a
                          className="text-white underline hover:text-neutral-300"
                          {...props}
                        />
                      ),
                      strong: ({ node, ...props }) => (
                        <strong className="font-bold text-white" {...props} />
                      ),
                      em: ({ node, ...props }) => (
                        <em className="italic text-neutral-300" {...props} />
                      ),
                      hr: ({ node, ...props }) => (
                        <hr className="border-white/10 my-8" {...props} />
                      ),
                      table: ({ node, ...props }) => (
                        <div className="overflow-x-auto mb-4">
                          <table
                            className="min-w-full border border-white/10"
                            {...props}
                          />
                        </div>
                      ),
                      thead: ({ node, ...props }) => (
                        <thead className="bg-neutral-900" {...props} />
                      ),
                      tbody: ({ node, ...props }) => <tbody {...props} />,
                      tr: ({ node, ...props }) => (
                        <tr className="border-b border-white/10" {...props} />
                      ),
                      th: ({ node, ...props }) => (
                        <th
                          className="px-4 py-2 text-left text-white font-bold"
                          {...props}
                        />
                      ),
                      td: ({ node, ...props }) => (
                        <td className="px-4 py-2 text-neutral-300" {...props} />
                      ),
                    }}
                  >
                    {readmeContent}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="py-6 border-t border-white/10 text-center text-neutral-600 text-xs font-mono">
          RT-DETR MODEL // D2-CITY DATASET // v1.0.4
        </footer>
      </div>
    </div>
  );
}

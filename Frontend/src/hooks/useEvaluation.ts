import { useState, useCallback } from "react";
import type { EvaluationResult, PipelineStep } from "@/types/evaluation";

const initialSteps: PipelineStep[] = [
  { id: "upload", name: "Image Upload", description: "Receiving answer sheet", status: "pending" },
  { id: "preprocess", name: "Preprocess Image", description: "Grayscale, blur, thresholding", status: "pending" },
  { id: "ocr", name: "OCR Extraction", description: "Vision API handwriting recognition", status: "pending" },
  { id: "nlp", name: "Text Preprocessing", description: "Tokenization, lemmatization", status: "pending" },
  { id: "embedding", name: "SBERT Embeddings", description: "Generate sentence vectors", status: "pending" },
  { id: "relevance", name: "Relevance Check", description: "Cosine similarity analysis", status: "pending" },
  { id: "semantic", name: "Semantic Analysis", description: "Logic flow evaluation", status: "pending" },
  { id: "nli", name: "Contradiction Detection", description: "NLI model inference", status: "pending" },
  { id: "scoring", name: "Final Scoring", description: "Weighted score calculation", status: "pending" },
];

// Backend API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_ENDPOINT = `${API_BASE_URL}/api/evaluate`;

// Backend evaluation - connects to FastAPI backend
async function callBackendEvaluation(
  question: string,
  referenceAnswer: string,
  imageFile: File,
  updateStep: (stepId: string, status: PipelineStep["status"]) => void
): Promise<EvaluationResult> {
  // Simulate pipeline steps for UI feedback
  const steps = ["upload", "preprocess", "ocr", "nlp", "embedding", "relevance", "semantic", "nli", "scoring"];
  
  // Start pipeline animation
  let currentStepIndex = 0;
  const stepInterval = setInterval(() => {
    if (currentStepIndex < steps.length) {
      updateStep(steps[currentStepIndex], "active");
      if (currentStepIndex > 0) {
        updateStep(steps[currentStepIndex - 1], "completed");
      }
      currentStepIndex++;
    }
  }, 400);

  try {
    // Prepare form data
    const formData = new FormData();
    formData.append("question", question);
    formData.append("reference_answer", referenceAnswer);
    formData.append("answer_image", imageFile);

    // Call backend API
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    clearInterval(stepInterval);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const result: EvaluationResult = await response.json();

    // Mark all steps as completed
    steps.forEach((step) => updateStep(step, "completed"));

    return result;
  } catch (error) {
    clearInterval(stepInterval);
    // Mark current step as error
    if (currentStepIndex < steps.length) {
      updateStep(steps[currentStepIndex], "error");
    }
    throw error;
  }
}

// Fallback simulated evaluation (for testing without backend)
async function simulateEvaluation(
  question: string,
  referenceAnswer: string,
  imageFile: File,
  updateStep: (stepId: string, status: PipelineStep["status"]) => void
): Promise<EvaluationResult> {
  const steps = ["upload", "preprocess", "ocr", "nlp", "embedding", "relevance", "semantic", "nli", "scoring"];
  
  for (const step of steps) {
    updateStep(step, "active");
    await new Promise((resolve) => setTimeout(resolve, 600 + Math.random() * 400));
    updateStep(step, "completed");
  }

  // Simulated result - in production, this comes from your API
  const similarity = 0.72 + Math.random() * 0.2;
  const logicFlow = 0.68 + Math.random() * 0.25;
  const contradiction = 0.75 + Math.random() * 0.2;
  
  const finalScore = 0.5 * similarity + 0.3 * logicFlow + 0.2 * contradiction;

  return {
    question_id: `Q-${Date.now()}`,
    student_raw_text: `[Simulated OCR Output]\n\nThe extracted handwritten text from the uploaded answer sheet would appear here. This includes the student's complete response as recognized by the Vision API OCR system.\n\nIn a production environment, this would contain the actual text extracted from: ${imageFile.name}`,
    cleaned_text: `The preprocessed and cleaned version of the student's answer after NLP processing (tokenization, stopword removal, lemmatization). This normalized text is used for embedding generation and semantic comparison.`,
    similarity_score: Math.min(similarity, 1),
    logic_flow_score: Math.min(logicFlow, 1),
    contradiction_score: Math.min(contradiction, 1),
    final_score: Math.min(finalScore, 1),
    relevance_flag: similarity > 0.5 ? "relevant" : "irrelevant",
  };
}

export function useEvaluation() {
  const [steps, setSteps] = useState<PipelineStep[]>(initialSteps);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateStep = useCallback((stepId: string, status: PipelineStep["status"]) => {
    setSteps((prev) =>
      prev.map((step) => (step.id === stepId ? { ...step, status } : step))
    );
  }, []);

  const resetSteps = useCallback(() => {
    setSteps(initialSteps);
    setResult(null);
    setError(null);
  }, []);

  const evaluate = useCallback(
    async (data: { question: string; referenceAnswer: string; imageFile: File }) => {
      setIsLoading(true);
      setError(null);
      resetSteps();

      try {
        // Only try real backend - no simulation fallback
        const evaluationResult = await callBackendEvaluation(
          data.question,
          data.referenceAnswer,
          data.imageFile,
          updateStep
        );
        
        setResult(evaluationResult);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Evaluation failed";
        
        // Check if it's a network error (backend not running)
        if (errorMessage.includes("fetch") || errorMessage.includes("NetworkError")) {
          setError("❌ Backend server is not running. Please start the Python backend server on http://localhost:8000");
        } else {
          setError(errorMessage);
        }
        
        // Mark current step as error
        setSteps((prev) => {
          const activeStep = prev.find((s) => s.status === "active");
          if (activeStep) {
            return prev.map((s) =>
              s.id === activeStep.id ? { ...s, status: "error" } : s
            );
          }
          return prev;
        });
      } finally {
        setIsLoading(false);
      }
    },
    [resetSteps, updateStep]
  );

  return {
    steps,
    result,
    isLoading,
    error,
    evaluate,
    resetSteps,
  };
}

import { BarChart3, GitBranch, Scale, Target } from "lucide-react";
import { ScoreCard } from "./ScoreCard";
import { FinalVerdict } from "./FinalVerdict";
import type { EvaluationResult } from "@/types/evaluation";

interface ResultsDisplayProps {
  result: EvaluationResult;
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const extractedText = result.cleaned_text || result.student_raw_text || "No text extracted";
  const hasRawText = result.student_raw_text && result.student_raw_text !== result.cleaned_text;
  
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Extracted Text */}
      <div className="rounded-xl border border-border bg-card p-6 shadow-card">
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Extracted Student Answer
        </h3>
        <div className="text-extracted max-h-[200px] overflow-y-auto whitespace-pre-wrap">
          {extractedText}
        </div>
        
        {/* Show raw text if different from cleaned */}
        {hasRawText && (
          <details className="mt-4">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              View Raw OCR Output
            </summary>
            <div className="mt-2 rounded-lg bg-muted/30 p-3 font-mono text-xs whitespace-pre-wrap max-h-[150px] overflow-y-auto">
              {result.student_raw_text}
            </div>
          </details>
        )}
        
        {result.relevance_flag === "irrelevant" && (
          <div className="mt-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            ⚠️ This answer was flagged as potentially irrelevant to the question.
          </div>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ScoreCard
          title="Similarity Score"
          score={result.similarity_score}
          icon={BarChart3}
          description="Semantic similarity to reference answer"
        />
        <ScoreCard
          title="Logic Flow"
          score={result.logic_flow_score}
          icon={GitBranch}
          description="Coherence and structural consistency"
        />
        <ScoreCard
          title="Contradiction Check"
          score={result.contradiction_score}
          icon={Scale}
          description="Factual accuracy via NLI analysis"
        />
        <ScoreCard
          title="Relevance"
          score={result.relevance_flag === "relevant" ? 1 : 0.2}
          icon={Target}
          description="Answer relevance to the question"
        />
      </div>

      <FinalVerdict score={result.final_score} />
    </div>
  );
}

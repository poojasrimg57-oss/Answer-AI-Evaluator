import { Check, Loader2, Circle, Image, FileText, Brain, Search, GitBranch, Scale, Calculator } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PipelineStep } from "@/types/evaluation";

const stepIcons: Record<string, React.ElementType> = {
  upload: Image,
  preprocess: FileText,
  ocr: FileText,
  nlp: Brain,
  embedding: Brain,
  relevance: Search,
  semantic: GitBranch,
  nli: Scale,
  scoring: Calculator,
};

interface PipelineVisualizationProps {
  steps: PipelineStep[];
}

export function PipelineVisualization({ steps }: PipelineVisualizationProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-card">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        Evaluation Pipeline
      </h3>
      <div className="space-y-1">
        {steps.map((step, index) => {
          const Icon = stepIcons[step.id] || Circle;
          return (
            <div key={step.id} className="relative">
              <div
                className={cn(
                  "pipeline-step",
                  step.status === "active" && "active",
                  step.status === "completed" && "completed"
                )}
              >
                <div
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full border-2 transition-all duration-300",
                    step.status === "pending" && "border-muted-foreground/30 text-muted-foreground/50",
                    step.status === "active" && "border-primary bg-primary text-primary-foreground",
                    step.status === "completed" && "border-accent bg-accent text-accent-foreground",
                    step.status === "error" && "border-destructive bg-destructive text-destructive-foreground"
                  )}
                >
                  {step.status === "completed" ? (
                    <Check className="h-4 w-4" />
                  ) : step.status === "active" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                </div>
                <div className="flex-1">
                  <p
                    className={cn(
                      "text-sm font-medium",
                      step.status === "pending" && "text-muted-foreground",
                      step.status === "active" && "text-primary",
                      step.status === "completed" && "text-accent"
                    )}
                  >
                    {step.name}
                  </p>
                  <p className="text-xs text-muted-foreground">{step.description}</p>
                </div>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={cn(
                    "absolute left-[19px] top-[44px] h-3 w-0.5 transition-colors duration-300",
                    step.status === "completed" ? "bg-accent" : "bg-border"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

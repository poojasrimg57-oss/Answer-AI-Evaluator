import { Header } from "@/components/Header";
import { EvaluationForm } from "@/components/EvaluationForm";
import { PipelineVisualization } from "@/components/PipelineVisualization";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useEvaluation } from "@/hooks/useEvaluation";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";

const Index = () => {
  const { steps, result, isLoading, error, evaluate, resetSteps } = useEvaluation();

  const hasStarted = steps.some((s) => s.status !== "pending");

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container py-8">
        {/* Hero Section */}
        <section className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-bold tracking-tight text-foreground md:text-5xl">
            AI-Powered Answer Sheet
            <span className="block bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
              Evaluation System
            </span>
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
            Upload handwritten answer sheets and get instant, intelligent grading using 
            OCR, SBERT embeddings, semantic analysis, and NLI contradiction detection.
          </p>
        </section>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Left Column - Form */}
          <div className="lg:col-span-2">
            <div className="rounded-xl border border-border bg-card p-6 shadow-card">
              <div className="mb-6 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Submit for Evaluation</h2>
                {hasStarted && (
                  <Button variant="ghost" size="sm" onClick={resetSteps}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Reset
                  </Button>
                )}
              </div>
              <EvaluationForm onSubmit={evaluate} isLoading={isLoading} />
            </div>

            {/* Results Section */}
            {result && (
              <div className="mt-8">
                <h2 className="mb-6 text-xl font-semibold">Evaluation Results</h2>
                <ResultsDisplay result={result} />
              </div>
            )}

            {error && (
              <div className="mt-8 rounded-xl border border-destructive bg-destructive/10 p-6">
                <h3 className="font-semibold text-destructive">Evaluation Error</h3>
                <p className="mt-2 text-sm text-destructive/80">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Pipeline */}
          <div className="space-y-6">
            <PipelineVisualization steps={steps} />
            
            {/* Info Card */}
            <div className="rounded-xl border border-border bg-card p-6 shadow-card">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                Scoring Weights
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Similarity Score</span>
                  <span className="font-medium">50%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Logic Flow</span>
                  <span className="font-medium">30%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Contradiction</span>
                  <span className="font-medium">20%</span>
                </div>
              </div>
              <div className="mt-4 border-t border-border pt-4">
                <p className="text-xs text-muted-foreground">
                  Weights can be configured in the backend's <code className="rounded bg-muted px-1 font-mono">config.yaml</code>
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-8">
        <div className="container text-center text-sm text-muted-foreground">
          <p>AnswerAI Evaluator • Built with SBERT, Vision API, and NLI Models</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;

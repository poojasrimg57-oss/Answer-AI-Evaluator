import { Award, CheckCircle2, AlertCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { VerdictLevel } from "@/types/evaluation";

interface FinalVerdictProps {
  score: number;
}

function getVerdict(score: number): { level: VerdictLevel; label: string; description: string } {
  if (score >= 0.85) {
    return {
      level: "excellent",
      label: "Excellent",
      description: "Outstanding answer demonstrating deep understanding and accuracy.",
    };
  }
  if (score >= 0.7) {
    return {
      level: "good",
      label: "Good",
      description: "Solid answer with good comprehension and mostly accurate content.",
    };
  }
  if (score >= 0.5) {
    return {
      level: "needs-improvement",
      label: "Needs Improvement",
      description: "Partial understanding shown. Review key concepts and accuracy.",
    };
  }
  return {
    level: "poor",
    label: "Poor",
    description: "Significant gaps in understanding or accuracy issues detected.",
  };
}

const verdictStyles: Record<VerdictLevel, { bg: string; text: string; icon: typeof Award }> = {
  excellent: { bg: "bg-accent/10", text: "text-accent", icon: Award },
  good: { bg: "bg-yellow-500/10", text: "text-yellow-600", icon: CheckCircle2 },
  "needs-improvement": { bg: "bg-orange-500/10", text: "text-orange-600", icon: AlertCircle },
  poor: { bg: "bg-destructive/10", text: "text-destructive", icon: XCircle },
};

export function FinalVerdict({ score }: FinalVerdictProps) {
  const verdict = getVerdict(score);
  const styles = verdictStyles[verdict.level];
  const Icon = styles.icon;

  return (
    <div className={cn("rounded-xl border-2 p-6 text-center", styles.bg, `border-current ${styles.text}`)}>
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-background">
        <Icon className={cn("h-8 w-8", styles.text)} />
      </div>
      <div className={cn("mb-2 text-5xl font-bold tabular-nums", styles.text)}>
        {(score * 100).toFixed(1)}%
      </div>
      <h3 className={cn("mb-2 text-xl font-semibold", styles.text)}>{verdict.label}</h3>
      <p className="text-sm text-muted-foreground">{verdict.description}</p>
    </div>
  );
}

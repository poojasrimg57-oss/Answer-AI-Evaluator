import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface ScoreCardProps {
  title: string;
  score: number;
  maxScore?: number;
  icon: LucideIcon;
  description: string;
  className?: string;
}

function getScoreColor(score: number): string {
  if (score >= 0.8) return "text-accent";
  if (score >= 0.6) return "text-score-good";
  if (score >= 0.4) return "text-score-needsWork";
  return "text-destructive";
}

function getScoreBg(score: number): string {
  if (score >= 0.8) return "bg-accent/10";
  if (score >= 0.6) return "bg-yellow-500/10";
  if (score >= 0.4) return "bg-orange-500/10";
  return "bg-destructive/10";
}

export function ScoreCard({ title, score, maxScore = 1, icon: Icon, description, className }: ScoreCardProps) {
  const percentage = (score / maxScore) * 100;
  const scoreColor = getScoreColor(score);
  const scoreBg = getScoreBg(score);

  return (
    <div className={cn("score-card group", className)}>
      <div className="mb-4 flex items-start justify-between">
        <div className={cn("rounded-lg p-2", scoreBg)}>
          <Icon className={cn("h-5 w-5", scoreColor)} />
        </div>
        <div className="text-right">
          <span className={cn("text-3xl font-bold tabular-nums", scoreColor)}>
            {(score * 100).toFixed(0)}
          </span>
          <span className="text-lg text-muted-foreground">%</span>
        </div>
      </div>
      <h4 className="text-sm font-semibold text-foreground">{title}</h4>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all duration-500", 
            score >= 0.8 ? "bg-accent" :
            score >= 0.6 ? "bg-yellow-500" :
            score >= 0.4 ? "bg-orange-500" : "bg-destructive"
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

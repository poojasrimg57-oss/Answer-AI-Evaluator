import { useState, useCallback } from "react";
import { Upload, FileText, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface EvaluationFormProps {
  onSubmit: (data: { question: string; referenceAnswer: string; imageFile: File }) => void;
  isLoading: boolean;
}

export function EvaluationForm({ onSubmit, isLoading }: EvaluationFormProps) {
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/") || file.type === "application/pdf") {
        setImageFile(file);
        // Create preview for images
        if (file.type.startsWith("image/")) {
          const reader = new FileReader();
          reader.onloadend = () => {
            setImagePreview(reader.result as string);
          };
          reader.readAsDataURL(file);
        } else {
          setImagePreview(null);
        }
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      // Create preview for images
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        setImagePreview(null);
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question && referenceAnswer && imageFile) {
      onSubmit({ question, referenceAnswer, imageFile });
    }
  };

  const isValid = question.trim() && referenceAnswer.trim() && imageFile;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="question" className="text-sm font-medium">
          Question
        </Label>
        <Textarea
          id="question"
          placeholder="Enter the exam question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="min-h-[100px] resize-none"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="reference" className="text-sm font-medium">
          Reference Answer
        </Label>
        <Textarea
          id="reference"
          placeholder="Enter the model/reference answer for comparison..."
          value={referenceAnswer}
          onChange={(e) => setReferenceAnswer(e.target.value)}
          className="min-h-[150px] resize-none"
        />
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-medium">Student Answer Sheet</Label>
        <div
          className={cn(
            "relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-200",
            dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50",
            imageFile && "border-accent bg-accent/5"
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById("file-upload")?.click()}
        >
          <input
            id="file-upload"
            type="file"
            accept="image/*,.pdf"
            onChange={handleFileChange}
            className="hidden"
          />
          {imageFile ? (
            <div className="flex flex-col items-center gap-3 p-4 w-full">
              {imagePreview ? (
                <div className="w-full max-w-md">
                  <img 
                    src={imagePreview} 
                    alt="Answer sheet preview" 
                    className="w-full h-auto rounded-lg border-2 border-accent shadow-lg"
                  />
                </div>
              ) : (
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent/20">
                  <FileText className="h-6 w-6 text-accent" />
                </div>
              )}
              <div className="text-center">
                <p className="font-medium text-foreground">{imageFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(imageFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setImageFile(null);
                  setImagePreview(null);
                }}
                className="text-muted-foreground hover:text-destructive"
              >
                <X className="mr-1 h-4 w-4" />
                Remove
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 p-6 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-muted">
                <Upload className="h-6 w-6 text-muted-foreground" />
              </div>
              <div>
                <p className="font-medium text-foreground">Drop answer sheet here</p>
                <p className="text-sm text-muted-foreground">or click to browse</p>
              </div>
              <p className="text-xs text-muted-foreground">Supports: JPG, PNG, PDF</p>
            </div>
          )}
        </div>
      </div>

      <Button
        type="submit"
        variant="hero"
        size="lg"
        className="w-full"
        disabled={!isValid || isLoading}
      >
        {isLoading ? (
          <>
            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
            Evaluating...
          </>
        ) : (
          <>
            <Send className="mr-2 h-4 w-4" />
            Evaluate Answer
          </>
        )}
      </Button>
    </form>
  );
}

export interface EvaluationResult {
  question_id: string;
  student_raw_text: string;
  cleaned_text: string;
  similarity_score: number;
  logic_flow_score: number;
  contradiction_score: number;
  final_score: number;
  relevance_flag: 'relevant' | 'irrelevant';
}

export interface PipelineStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'error';
}

export type VerdictLevel = 'excellent' | 'good' | 'needs-improvement' | 'poor';

export interface EvaluationRequest {
  question: string;
  reference_answer: string;
  answer_image: File;
}

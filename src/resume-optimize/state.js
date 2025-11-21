import { Annotation } from '@langchain/langgraph'

// Define the state structure for resume optimization
export const ResumeState = Annotation.Root({
  resumeText: Annotation({ reducer: (_, x) => x }),
  personalInfo: Annotation({ reducer: (_, x) => x }),
  skills: Annotation({ reducer: (_, x) => x }),
  projects: Annotation({ reducer: (_, x) => x }),
  yearsOfExperience: Annotation({ reducer: (_, x) => x }),
  skillsEvaluation: Annotation({ reducer: (_, x) => x }),
  projectsEvaluation: Annotation({ reducer: (_, x) => x }),
  suggestions: Annotation({ reducer: (_, x) => x }),
})

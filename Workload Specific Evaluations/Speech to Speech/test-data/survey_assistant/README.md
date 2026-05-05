# Customer Survey Assistant Test Scenario

## Overview

This scenario tests a conversational AI assistant that conducts voice-based customer satisfaction surveys about Amazon Nova Sonic. The assistant gathers structured feedback through natural conversation while maintaining a friendly, non-judgmental tone.

## Purpose

- **Primary Goal:** Evaluate survey conversation management and feedback collection
- **Key Behaviors:** Question sequencing, active listening, neutral acknowledgment, follow-up prompting
- **Use Case:** Voice-based customer feedback and satisfaction surveys

## Scenario Details

### Profile Configuration
- **Voice Model:** Nova Sonic v1 - Legacy
- **Voice ID:** Matthew (friendly, professional voice)
- **Conversation Style:** Conversational survey with structured questions
- **Expected Duration:** 2-3 minutes (7 interaction turns)

### System Prompt Highlights

The assistant (SurveyBot) is configured to:
- Conduct a brief, natural-sounding voice survey
- Keep responses short (2-3 sentences)
- Ask three main questions: rating, positive feedback, improvement suggestions
- Maintain friendly, non-defensive tone
- Accept all feedback neutrally and appreciatively

### Survey Questions

1. **Overall Rating:** "On a scale of 1 to 5, with 5 being the best, how would you rate your experience with Amazon Nova Sonic?"
2. **Positive Feedback:** "Great, thank you! What do you love most about the model? What stands out to you?"
3. **Improvement Areas:** "And finally, what would you like to see improved, or is there anything you think is missing?"

## Audio Files

The scenario includes 7 pre-recorded audio files representing a complete survey conversation:

- `01_Hello.wav` - Initial greeting response
- `02_HappyToHelp.wav` - Willingness to participate
- `03_Give4.wav` - Rating response (4 out of 5)
- `04_ReallyNormal.wav` - Positive feedback about naturalness
- `05_HandlesAccents.wav` - Additional positive feedback
- `06_PausesOdd.wav` - Improvement suggestion about pauses
- `07_Welcome.wav` - Closing acknowledgment

## Expected Conversation Flow

1. **Opening:** Assistant greets and explains survey purpose
2. **Question 1:** Rating question → Customer provides rating (1-5)
3. **Acknowledgment:** Assistant acknowledges rating positively
4. **Question 2:** What they love most → Customer shares positive feedback
5. **Follow-up:** Assistant may prompt for more details
6. **Question 3:** Improvement suggestions → Customer provides feedback
7. **Closing:** Thank you and wrap-up

## Success Criteria

### Speech Recognition
- Accurately transcribe ratings (numbers 1-5)
- Understand feedback terminology (natural, accents, pauses, etc.)
- Handle casual speech patterns

### Conversation Quality
- Maintain neutral, appreciative tone regardless of feedback
- Acknowledge all responses positively
- Smooth transitions between questions
- Give users time to think and respond

### Survey Management
- Ask all three questions in order
- Don't skip questions even if user provides multiple answers
- Handle "nothing" or "no suggestions" responses gracefully
- Prompt for more details when responses are brief

### Professional Tone
- Non-defensive about negative feedback
- Genuinely appreciative of all input
- Friendly but not overly casual
- Validate all feedback as valuable

## Evaluation Metrics

- **Speech Recognition Accuracy:** 95%+ for ratings and feedback terms
- **Tool Calling Accuracy:** N/A (no tools configured for this scenario)
- **Response Relevance:** 100% appropriate survey responses
- **Context Maintenance:** Remember previous answers, don't repeat questions
- **Completeness:** All three questions asked and answered
- **Clarity:** Clear, concise, friendly survey questions

## Difficulty Level

**Easy** - Straightforward linear question flow with predictable structure, but requires maintaining neutral tone and handling varied feedback.

## Sample Manual Test Conversation

If testing manually through the UI:

1. Click "Start Conversation"
2. When greeted, say: "Hello, I'm happy to help with the survey"
3. When asked for rating, say: "I'd give it a 4 out of 5"
4. When asked what you love, say: "The voice sounds really natural and conversational"
5. Add more if prompted: "It handles different accents well"
6. When asked about improvements, say: "Sometimes the pauses between responses feel a bit odd"
7. Respond to closing: "You're welcome, happy to help"

## Notes

- This scenario tests structured conversation flow with open-ended responses
- Focus is on maintaining neutral, appreciative tone regardless of feedback
- Assistant should never argue with or challenge user feedback
- Brief survey format (under 2 minutes) tests efficiency
- No technical support should be provided during the survey

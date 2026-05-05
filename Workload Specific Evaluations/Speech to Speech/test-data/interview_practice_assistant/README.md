# Interview Practice Assistant Test Scenario

## Overview

This scenario tests an AI-powered interview practice assistant that conducts technical interviews using Amazon Nova Sonic's speech-to-speech capabilities. The assistant helps candidates prepare for technical roles by simulating realistic interview conversations.

## Purpose

- **Primary Goal:** Evaluate the assistant's ability to conduct structured technical interviews
- **Key Behaviors:** Question sequencing, STAR format evaluation, follow-up questioning, context maintenance
- **Use Case:** Interview preparation for cloud architecture and technical leadership roles

## Scenario Details

### Profile Configuration
- **Voice Model:** Nova Sonic v1 - Legacy
- **Voice ID:** Matthew (professional, clear voice)
- **Conversation Style:** Structured interview with turn-taking
- **Expected Duration:** 5-8 minutes (5 questions + follow-ups)

### System Prompt Highlights

The assistant is configured to:
- Follow strict turn-taking (one question at a time)
- Evaluate responses using Amazon's STAR format (Situation, Task, Action, Result)
- Ask targeted follow-up questions for incomplete answers
- Maintain professional yet conversational tone
- Recognize meta-communication vs. substantive answers

### Interview Questions

1. **Background Question:** "Can you tell me about your background and what led you to pursue a career in cloud architecture?"
2. **Project Experience:** "Can you describe a complex cloud architecture project you've worked on and the challenges you faced?"
3. **Technical Process:** "How do you approach designing a scalable cloud data platform? Can you walk me through your process?"
4. **Migration Scenario:** "Imagine you are tasked with migrating a legacy system to the cloud. What steps would you take to ensure a smooth transition?"
5. **Candidate Questions:** "What questions do you have for us about the role or the company?"

## Audio Files

The scenario includes 5 pre-recorded audio files representing candidate responses:

- `01_TellMeAboutYourself.wav` - Background and career motivation
- `02_CanYouDescribeAComplexCloudproject.wav` - Project experience overview
- `03_CloudMigrationProject.wav` - Detailed migration example
- `04_HowHaveYouLedTechnicalTeams.wav` - Leadership experience
- `05_WhatExcitesYouAboutThisRole.wav` - Candidate questions and interest

## Expected Conversation Flow

1. **Opening:** Assistant greets and explains interview format
2. **Question 1:** Background question → Candidate responds → Assistant evaluates
3. **Follow-ups:** If answer incomplete, assistant asks targeted follow-up questions
4. **Question 2-4:** Technical questions with STAR evaluation
5. **Closing:** Final questions from candidate, thank you and wrap-up

## Success Criteria

### Speech Recognition
- Accurately transcribe technical terminology (cloud, architecture, migration, etc.)
- Handle pauses and thinking time appropriately

### Conversation Quality
- Maintain strict turn-taking (no interruptions)
- Ask relevant follow-up questions for incomplete STAR answers
- Maintain context across multiple turns
- Recognize when to move to next question

### STAR Format Evaluation
- Identify missing STAR components (Situation, Task, Action, Result)
- Ask targeted questions to elicit missing information
- Recognize complete vs. incomplete answers

### Professional Tone
- Maintain encouraging yet objective assessment
- Avoid repetitive phrases
- Adapt language to conversation flow

## Evaluation Metrics

- **Speech Recognition Accuracy:** 95%+ for technical terms
- **Tool Calling Accuracy:** Correct use of `get_next_interview_question` tool
- **Response Relevance:** 100% on-topic responses
- **Context Maintenance:** No context loss across turns
- **Completeness:** All questions asked, proper follow-ups
- **Clarity:** Professional, well-structured responses

## Difficulty Level

**Medium-High** - Requires sophisticated conversation management, STAR evaluation logic, and adaptive follow-up questioning.

## Sample Manual Test Conversation

If testing manually through the UI:

1. Click "Start Conversation"
2. Say: "I've been working in cloud architecture for about 5 years, starting with AWS infrastructure design."
3. Wait for follow-up question
4. Describe a specific project with challenges
5. Continue answering questions naturally

## Notes

- The assistant should NOT read special characters (bullets, arrows, etc.)
- Silence and thinking time are natural - assistant should wait patiently
- Meta-communication (e.g., "Can I have a moment to think?") should be acknowledged differently than substantive answers

class S2sEvent {
    static DEFAULT_INFER_CONFIG = {
      maxTokens: 1024,
      topP: 0.95,
      temperature: 0.7
    };
  
    static DEFAULT_SYSTEM_PROMPT = "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios.";
    
    static DEFAULT_SENTIMENT_PROMPT = "Convert streaming speech to text, call an external model for text responses, emit turn taking events, and convert the text to speech. The transcription and response should be in spoken form with no capitalization and punctuations. Add per turn user emotion to the output.";

    static DEFAULT_AUDIO_INPUT_CONFIG = {
      mediaType: "audio/lpcm",
      sampleRateHertz: 16000,
      sampleSizeBits: 16,
      channelCount: 1,
      audioType: "SPEECH",
      encoding: "base64"
    };
  
    static DEFAULT_AUDIO_OUTPUT_CONFIG = {
      mediaType: "audio/lpcm",
      sampleRateHertz: 24000,
      sampleSizeBits: 16,
      channelCount: 1,
      voiceId: "matthew",
      encoding: "base64",
      audioType: "SPEECH"
    };
  
    static DEFAULT_TOOL_CONFIG = {
      tools: [
    ]
    };

    static DEFAULT_CHAT_HISTORY = [
      {}
    ];
  
    static sessionStart(inferenceConfig = S2sEvent.DEFAULT_INFER_CONFIG, turnSensitivity="LOW") {
      return { 
        event: { 
          sessionStart: { 
            inferenceConfiguration: inferenceConfig ,
            turnDetectionConfiguration: {
              endpointingSensitivity: turnSensitivity
            }
          }
        } 
      };
    }
  
    static promptStart(promptName, audioOutputConfig = S2sEvent.DEFAULT_AUDIO_OUTPUT_CONFIG, toolConfig = S2sEvent.DEFAULT_TOOL_CONFIG) {
      return {
        "event": {
          "promptStart": {
            "promptName": promptName,
            "textOutputConfiguration": {
              "mediaType": "text/plain"
            },
            "audioOutputConfiguration": audioOutputConfig,
          
          "toolUseOutputConfiguration": {
            "mediaType": "application/json"
          },
          "toolConfiguration": toolConfig
        }
        }
      }
    }
  
    static contentStartText(promptName, contentName, role="SYSTEM", interactive=false) {
      return {
        "event": {
          "contentStart": {
            "promptName": promptName,
            "contentName": contentName,
            "type": "TEXT",
            "interactive": interactive,
            "role": role,
            "textInputConfiguration": {
              "mediaType": "text/plain"
            }
          }
        }
      }
    }
  
    static textInput(promptName, contentName, systemPrompt = S2sEvent.DEFAULT_SYSTEM_PROMPT) {
      var evt = {
        "event": {
          "textInput": {
            "promptName": promptName,
            "contentName": contentName,
            "content": systemPrompt
          }
        }
      }
      return evt;
    }
  
    static contentEnd(promptName, contentName) {
      return {
        "event": {
          "contentEnd": {
            "promptName": promptName,
            "contentName": contentName
          }
        }
      }
    }
  
    static contentStartAudio(promptName, contentName, audioInputConfig = S2sEvent.DEFAULT_AUDIO_INPUT_CONFIG) {
      return {
        "event": {
          "contentStart": {
            "promptName": promptName,
            "contentName": contentName,
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
              "mediaType": "audio/lpcm",
              "sampleRateHertz": 16000,
              "sampleSizeBits": 16,
              "channelCount": 1,
              "audioType": "SPEECH",
              "encoding": "base64"
            }
          }
        }
      }
    }
  
    static audioInput(promptName, contentName, content) {
      return {
        event: {
          audioInput: {
            promptName,
            contentName,
            content,
          }
        }
      };
    }
  
    static contentStartTool(promptName, contentName, toolUseId) {
      return {
        event: {
          contentStart: {
            promptName,
            contentName,
            interactive: false,
            type: "TOOL",
            toolResultInputConfiguration: {
              toolUseId,
              type: "TEXT",
              textInputConfiguration: { mediaType: "text/plain" }
            }
          }
        }
      };
    }
  
    static textInputTool(promptName, contentName, content) {
      return {
        event: {
          textInput: {
            promptName,
            contentName,
            content,
            role: "TOOL"
          }
        }
      };
    }
  
    static promptEnd(promptName) {
      return {
        event: {
          promptEnd: {
            promptName
          }
        }
      };
    }
  
    static sessionEnd() {
      return { event: { sessionEnd: {} } };
    }
  }
  export default S2sEvent;
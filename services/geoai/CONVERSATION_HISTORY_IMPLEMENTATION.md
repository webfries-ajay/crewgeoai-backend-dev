# Conversation History Implementation

## Overview

The GeoAI system now includes persistent conversation history using the latest LangChain patterns. Each project maintains its own conversation thread per user, allowing for contextual, multi-turn conversations that persist across sessions.

## Key Features

### 🔄 Persistent Storage
- Conversations are stored in JSON files under `conversation_history/{user_id}/{project_id}/`
- History persists across server restarts and user sessions
- Automatic file-based backup and recovery

### 🧠 Intelligent Memory Management
- Uses LangChain's `ConversationSummaryBufferMemory` with 2000 token limit
- Automatically summarizes old conversations to maintain context while staying within token limits
- Fallback to basic memory if summary features are unavailable

### 🎯 Project-Based Conversations
- Each project has its own conversation thread per user
- Category-specific master prompts are maintained throughout conversations
- Conversation context includes project category and user preferences

### 📊 Conversation Analytics
- Track message counts, conversation duration, and activity patterns
- Get detailed statistics about user engagement
- Monitor conversation health and performance

## Architecture

### Core Components

1. **`ProjectChatHistory`**: Custom chat history implementation
   - Extends LangChain's `BaseChatMessageHistory`
   - Handles file-based persistence
   - Manages message serialization/deserialization

2. **`ConversationManager`**: Central conversation management
   - Manages multiple conversation threads
   - Handles memory allocation and cleanup
   - Provides conversation statistics and analytics

3. **Integration Points**:
   - `UnifiedLangChainAnalyzer`: Uses conversation context for analysis
   - `SmartGeoAIAgent`: Maintains conversation state across interactions
   - `files.py` router: Provides conversation API endpoints

### File Structure
```
conversation_history/
├── {user_id}/
│   └── {project_id}/
│       └── chat_history.json
```

### Message Format
```json
{
  "project_id": "uuid",
  "user_id": "uuid", 
  "last_updated": "2024-01-01T12:00:00",
  "messages": [
    {
      "type": "human|ai|system",
      "content": "message content",
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

## API Endpoints

### Chat with File/Project
```http
POST /files/{file_id}/chat
{
  "message": "Your question",
  "projectId": "project-uuid"
}
```

### Get Conversation History
```http
GET /files/{file_id}/chat/history
```

### Reset Conversation
```http
POST /files/{file_id}/chat/reset
```

## Usage Examples

### Basic Chat Flow
```python
from services.geoai.smart_geoai_agent import SmartGeoAIAgent

# Create agent with conversation context
agent = SmartGeoAIAgent(
    master_prompt="You are a mining analysis specialist",
    project_id="project-123",
    user_id="user-456"
)

# Process messages - history is maintained automatically
response1 = agent.process_message("Analyze this mining survey image")
response2 = agent.process_message("What were the key findings from the previous analysis?")
# Agent remembers the previous analysis context
```

### Direct Conversation Management
```python
from services.geoai.conversation_manager import conversation_manager

# Add messages manually
conversation_manager.add_user_message("project-123", "user-456", "Hello")
conversation_manager.add_ai_message("project-123", "user-456", "Hi there!")

# Get history
history = conversation_manager.get_conversation_history("project-123", "user-456")

# Get statistics
stats = conversation_manager.get_conversation_stats("project-123", "user-456")

# Clear conversation
conversation_manager.clear_conversation("project-123", "user-456")
```

## Benefits

### For Users
- **Contextual Conversations**: AI remembers previous interactions within each project
- **Seamless Experience**: Conversation continues across sessions and page refreshes
- **Project-Specific Context**: Each project maintains its own conversation thread
- **Category Awareness**: AI maintains expertise context throughout conversations

### For Developers
- **Easy Integration**: Simple API for adding conversation features
- **Scalable Architecture**: Efficient memory management and cleanup
- **Robust Persistence**: File-based storage with error handling
- **Analytics Ready**: Built-in conversation tracking and statistics

## Performance Considerations

### Memory Management
- Conversations are loaded on-demand
- Automatic cleanup of inactive conversations (24-hour default)
- Token-based memory limits prevent excessive context growth
- Efficient file-based storage with minimal memory footprint

### Scalability
- File-based storage scales with filesystem capacity
- Per-user/per-project isolation prevents cross-contamination
- Background cleanup processes maintain system health
- Configurable limits and thresholds

## Configuration

### Environment Variables
```env
# Conversation settings (optional)
CONVERSATION_MAX_TOKEN_LIMIT=2000
CONVERSATION_CLEANUP_HOURS=24
CONVERSATION_STORAGE_PATH=conversation_history
```

### Memory Settings
- `max_token_limit`: Maximum tokens to keep in active memory (default: 2000)
- `cleanup_hours`: Hours before inactive conversations are cleaned up (default: 24)
- `storage_path`: Directory for conversation storage (default: "conversation_history")

## Troubleshooting

### Common Issues

1. **"No conversation history"**: Check file permissions in conversation_history directory
2. **"Memory errors"**: Verify OpenAI API key for summary features
3. **"Persistence issues"**: Ensure write permissions to storage directory

### Debug Information
The system provides detailed logging:
- Conversation creation and loading
- Message addition and retrieval
- Memory management operations
- File system operations

### Recovery
- Conversations can be manually restored from JSON files
- Corrupted files are automatically handled with graceful fallback
- Missing directories are created automatically

## Future Enhancements

### Planned Features
- **Database Storage**: Optional database backend for enterprise deployments
- **Conversation Branching**: Support for multiple conversation threads per project
- **Export/Import**: Conversation backup and migration tools
- **Advanced Analytics**: Detailed conversation insights and reporting
- **Real-time Sync**: WebSocket-based conversation synchronization

### Integration Opportunities
- **Multi-modal History**: Support for image and file references in history
- **Collaboration Features**: Shared conversations for team projects
- **AI Training**: Use conversation data for model fine-tuning
- **Search Integration**: Full-text search across conversation history

## Migration Guide

### From Previous System
The new system is backward compatible. Existing projects will automatically get conversation history enabled on first interaction.

### Manual Migration
If you have existing conversation data:
1. Create conversation_history directory structure
2. Convert existing data to JSON format
3. Place files in appropriate user/project directories
4. Restart the application

## Testing

The implementation includes comprehensive tests covering:
- Conversation creation and persistence
- Message addition and retrieval
- Memory management and cleanup
- Error handling and recovery
- Performance under load

Run tests with:
```bash
python test_conversation_history.py
```

## Security Considerations

### Data Protection
- Conversations are stored per-user with proper isolation
- File permissions restrict access to conversation data
- No sensitive information is logged in conversation files

### Privacy
- Conversations are only accessible to the project owner
- No cross-user data leakage
- Automatic cleanup prevents long-term data accumulation

### Compliance
- GDPR-ready with conversation deletion capabilities
- Audit trail for conversation access and modifications
- Configurable data retention policies 